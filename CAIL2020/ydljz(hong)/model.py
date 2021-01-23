import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
from transformers import BertModel


from resnet import ResNet,BasicBlock
from resnet2d import ResNet2D
from resnet2d import BasicBlock as BasicBlock2D

from capsnetx import PrimaryCaps, FCCaps, FlattenCaps


class BertSupportNetX(nn.Module):
    """
    joint train bert and graph fusion net
    """
    def __init__(self, config):
        super(BertSupportNetX, self).__init__()

        self.encoder = BertModel.from_pretrained(config.bert_model_path)
        self.config = config  # 就是args
        self.max_query_length = self.config.max_query_len
        self.input_dim = config.hidden_size
        self.dropout_size = config.dropout
        self.dropout = nn.Dropout(self.dropout_size)

        self.dim_capsule = config.dim_capsule
        self.num_compressed_capsule = config.num_compressed_capsule
        self.ngram_size = [2, 4, 8]
        self.convs_doc = nn.ModuleList([nn.Conv1d(config.max_seq_len, 32, K, stride=2) for K in self.ngram_size])
        torch.nn.init.xavier_uniform_(self.convs_doc[0].weight)
        torch.nn.init.xavier_uniform_(self.convs_doc[1].weight)
        torch.nn.init.xavier_uniform_(self.convs_doc[2].weight)

        self.primary_capsules_doc = PrimaryCaps(num_capsules=self.dim_capsule, in_channels=32, out_channels=32,
                                                kernel_size=1, stride=1)

        self.flatten_capsules = FlattenCaps()

        self.W_doc = nn.Parameter(torch.FloatTensor(config.magic_param, self.num_compressed_capsule))
        torch.nn.init.xavier_uniform_(self.W_doc)

        self.fc_capsules_doc_child = FCCaps(config, output_capsule_num=config.num_classes,
                                            input_capsule_num=self.num_compressed_capsule,
                                            in_channels=self.dim_capsule, out_channels=self.dim_capsule)

        self.start_linear = nn.Linear(self.input_dim*2, 1)
        self.end_linear = nn.Linear(self.input_dim*2, 1)
        self.type_linear = nn.Linear(self.input_dim, config.num_classes)  # yes/no/ans/unknown
        self.sp_linear = nn.Linear(self.input_dim, 1)

        self.cache_S = 0
        self.cache_mask = None

    def compression(self, poses, W):
        poses = torch.matmul(poses.permute(0, 2, 1), W).permute(0, 2, 1)
        activations = torch.sqrt((poses ** 2).sum(2))
        return poses, activations


    def get_output_mask(self, outer):
        # (batch, 512, 512)
        S = outer.size(1)
        if S <= self.cache_S:
            return Variable(self.cache_mask[:S, :S], requires_grad=False)
        self.cache_S = S
        # triu 生成上三角矩阵，tril生成下三角矩阵，这个相当于生成了(512, 512)的矩阵表示开始-结束的位置，答案长度最长为15
        np_mask = np.tril(np.triu(np.ones((S, S)), 0), 15)
        self.cache_mask = outer.data.new(S, S).copy_(torch.from_numpy(np_mask))
        return Variable(self.cache_mask, requires_grad=False)

    def forward(self, context_idxs, context_mask, segment_idxs,
            query_mapping, all_mapping,
            ids, y1, y2, q_type,
            start_mapping,
            is_support,tok_to_orig_index):
        # roberta不可以输入token_type_ids
        hiddens = self.encoder(input_ids=context_idxs, attention_mask=context_mask,token_type_ids=segment_idxs,
                                                                    output_hidden_states=True)[2]
        input_state = torch.cat([hiddens[-1], hiddens[-2]], dim=2)
        start_logits = self.start_linear(input_state).squeeze(2) - 1e30 * (1 - context_mask)
        end_logits = self.end_linear(input_state).squeeze(2) - 1e30 * (1 - context_mask)

        # type_state = torch.max(last_state, dim=1)[0]
        # type_logits = self.type_linear(type_state)
        nets_doc_l = []
        for i in range(len(self.ngram_size)):
            nets = self.convs_doc[i](hiddens[-1])
            nets_doc_l.append(nets)
        nets_doc = torch.cat((nets_doc_l[0], nets_doc_l[1], nets_doc_l[2]), 2)
        poses_doc, activations_doc = self.primary_capsules_doc(nets_doc)
        poses, activations = self.flatten_capsules(poses_doc, activations_doc)
        poses, activations = self.compression(poses, self.W_doc)
        poses, type_logits = self.fc_capsules_doc_child(poses, activations, range(4))#4 types in total.
        type_logits = type_logits.squeeze(2)

        last_state = hiddens[-1]
        sp_state = all_mapping.unsqueeze(3) * last_state.unsqueeze(2)  # N x sent x 512 x 300
        sp_state = sp_state.max(1)[0]
        sp_logits = self.sp_linear(sp_state)

        # 找结束位置用的开始和结束位置概率之和
        # (batch, 512, 1) + (batch, 1, 512) -> (512, 512)
        outer = start_logits[:, :, None] + end_logits[:, None]
        outer_mask = self.get_output_mask(outer)
        outer = outer - 1e30 * (1 - outer_mask[None].expand_as(outer))
        if query_mapping is not None:   # 这个是query_mapping (batch, 512)
            outer = outer - 1e30 * query_mapping[:, :, None]    # 不允许预测query的内容

        # 这两句相当于找到了outer中最大值的i和j坐标
        start_position = outer.max(dim=2)[0].max(dim=1)[1]
        end_position = outer.max(dim=1)[0].max(dim=1)[1]
        # return start_logits, end_logits, type_logits, sp_logits.squeeze(2), start_position, end_position
        return start_logits, end_logits, type_logits, sp_logits, start_position, end_position

class BertSupportNet(nn.Module):
    """
    joint train bert and graph fusion net
    """
    def __init__(self, config):
        super(BertSupportNet, self).__init__()
        # self.bert_model = BertModel.from_pretrained(config.bert_model)
        self.encoder = BertModel.from_pretrained(config.bert_model_path)
        self.graph_fusion_net = SupportNet(config)

    def forward(self, batch, debug=False):
        doc_ids, doc_mask, segment_ids = batch['context_idxs'], batch['context_mask'], batch['segment_idxs']
        # roberta不可以输入token_type_ids
        all_doc_encoder_layers = self.encoder(input_ids=doc_ids,
                                              token_type_ids=segment_ids,#可以注释
                                              attention_mask=doc_mask)[0]
        batch['context_encoding'] = all_doc_encoder_layers
        return self.graph_fusion_net(batch)


class SupportNet(nn.Module):
    """
    Packing Query Version
    """
    def __init__(self, config):
        super(SupportNet, self).__init__()
        self.config = config  # 就是args
        # self.n_layers = config.n_layers  # 2
        self.max_query_length = self.config.max_query_len
        self.prediction_layer = CNNPredictionLayer(config)

    def forward(self, batch, debug=False):
        context_encoding = batch['context_encoding']
        predictions = self.prediction_layer(batch, context_encoding)
        start_logits, end_logits, type_logits, sp_logits, start_position, end_position = predictions
        return start_logits, end_logits, type_logits, sp_logits, start_position, end_position


class SimplePredictionLayer(nn.Module):
    def __init__(self, config):
        super(SimplePredictionLayer, self).__init__()
        self.input_dim = config.input_dim
        self.sp_linear = nn.Linear(self.input_dim, 1)
        self.start_linear = nn.Linear(self.input_dim, 1)
        self.end_linear = nn.Linear(self.input_dim, 1)
        self.type_linear = nn.Linear(self.input_dim, config.label_type_num)   # yes/no/ans/unknown
        self.cache_S = 0
        self.cache_mask = None

    def get_output_mask(self, outer):
        # (batch, 512, 512)
        S = outer.size(1)
        if S <= self.cache_S:
            return Variable(self.cache_mask[:S, :S], requires_grad=False)
        self.cache_S = S
        # triu 生成上三角矩阵，tril生成下三角矩阵，这个相当于生成了(512, 512)的矩阵表示开始-结束的位置，答案长度最长为15
        np_mask = np.tril(np.triu(np.ones((S, S)), 0), 15)
        self.cache_mask = outer.data.new(S, S).copy_(torch.from_numpy(np_mask))
        return Variable(self.cache_mask, requires_grad=False)

    def forward(self, batch, input_state):
        query_mapping = batch['query_mapping']  # (batch, 512) 不一定是512，可能略小
        context_mask = batch['context_mask']  # bert里实际有输入的位置
        all_mapping = batch['all_mapping']  # (batch_size, 512, max_sent) 每个句子的token对应为1

        start_logits = self.start_linear(input_state).squeeze(2) - 1e30 * (1 - context_mask)
        end_logits = self.end_linear(input_state).squeeze(2) - 1e30 * (1 - context_mask)
        sp_state = all_mapping.unsqueeze(3) * input_state.unsqueeze(2)  # N x sent x 512 x 300
        sp_state = sp_state.max(1)[0]
        sp_logits = self.sp_linear(sp_state)
        type_state = torch.max(input_state, dim=1)[0]
        type_logits = self.type_linear(type_state)

        # 找结束位置用的开始和结束位置概率之和
        # (batch, 512, 1) + (batch, 1, 512) -> (512, 512)
        outer = start_logits[:, :, None] + end_logits[:, None]
        outer_mask = self.get_output_mask(outer)
        outer = outer - 1e30 * (1 - outer_mask[None].expand_as(outer))
        if query_mapping is not None:   # 这个是query_mapping (batch, 512)
            outer = outer - 1e30 * query_mapping[:, :, None]    # 不允许预测query的内容

        # 这两句相当于找到了outer中最大值的i和j坐标
        start_position = outer.max(dim=2)[0].max(dim=1)[1]
        end_position = outer.max(dim=1)[0].max(dim=1)[1]
        return start_logits, end_logits, type_logits, sp_logits.squeeze(2), start_position, end_position

import torch.nn.functional as F
class CNNPredictionLayer(nn.Module):
    def __init__(self, config):
        super(CNNPredictionLayer, self).__init__()
        self.input_dim = config.input_dim

        self.cnn_hidden_size = config.cnn_hidden_size
        self.cnn_output_size = config.cnn_output_size
        self.fc_hidden_size = config.fc_hidden_size
        self.dropout_size = config.dropout

        self.conv1 = nn.Conv1d(self.input_dim,  self.cnn_hidden_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(self.cnn_hidden_size, self.cnn_hidden_size, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(self.cnn_hidden_size, self.cnn_hidden_size, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(self.cnn_hidden_size, self.cnn_hidden_size, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(self.cnn_hidden_size, self.cnn_hidden_size, kernel_size=3, padding=1)
        self.conv6 = nn.Conv1d(self.cnn_hidden_size, self.fc_hidden_size, kernel_size=3, padding=1)

        # cnn feature map has a total number of 228 dimensions.
        # self.dropout = nn.Dropout(self.dropout_size)
        # self.fc1 = nn.Linear(config.cnn_output_size, config.fc_hidden_size)
        # self.fc2 = nn.Linear(config.cnn_output_size, config.fc_hidden_size)
        # self.fc3 = nn.Linear(config.fc_hidden_size, config.fc_hidden_size)
        # self.fc4 = nn.Linear(config.fc_hidden_size, config.fc_hidden_size)
        # self.fc5 = nn.Linear(config.fc_hidden_size, config.fc_hidden_size)

        self.sp_linear = nn.Linear(config.fc_hidden_size, 1)
        self.start_linear = nn.Linear(config.fc_hidden_size, 1)
        self.end_linear = nn.Linear(config.fc_hidden_size, 1)
        self.type_linear = nn.Linear(config.fc_hidden_size, config.label_type_num)   # yes/no/ans/unknown
        self.cache_S = 0
        self.cache_mask = None

    def get_output_mask(self, outer):
        # (batch, 512, 512)
        S = outer.size(1)
        if S <= self.cache_S:
            return Variable(self.cache_mask[:S, :S], requires_grad=False)
        self.cache_S = S
        # triu 生成上三角矩阵，tril生成下三角矩阵，这个相当于生成了(512, 512)的矩阵表示开始-结束的位置，答案长度最长为15
        np_mask = np.tril(np.triu(np.ones((S, S)), 0), 15)
        self.cache_mask = outer.data.new(S, S).copy_(torch.from_numpy(np_mask))
        return Variable(self.cache_mask, requires_grad=False)

    def forward(self, batch, input_state):
        query_mapping = batch['query_mapping']  # (batch, 512) 不一定是512，可能略小
        context_mask = batch['context_mask']  # bert里实际有输入的位置
        all_mapping = batch['all_mapping']  # (batch_size, 512, max_sent) 每个句子的token对应为1

        x = input_state.transpose(1, 2)#.type(torch.cuda.FloatTensor)
        x = F.max_pool1d(F.relu(self.conv1(x)), kernel_size=3, stride=1, padding=1)
        x = F.max_pool1d(F.relu(self.conv2(x)), kernel_size=3, stride=1, padding=1)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        input_state = x.transpose(2, 1)#.type(torch.cuda.FloatTensor)

        # x = F.max_pool1d(x, x.size(2)).squeeze(2)
        # x = F.relu(self.fc1(x.view(x.size(0), -1)))
        # x, y = self.fc1(cnn), self.fc2(cnn)
        # x, y = self.dropout(x), self.dropout(y)
        # x, y, z = self.fc3(x), self.fc4(y), self.fc5(cnn)
        # input_state, support_state, type_state = self.dropout(x), self.dropout(y), self.dropout(z)


        start_logits = self.start_linear(input_state).squeeze(2) - 1e30 * (1 - context_mask)
        end_logits = self.end_linear(input_state).squeeze(2) - 1e30 * (1 - context_mask)
        sp_state = all_mapping.unsqueeze(3) * input_state.unsqueeze(2)  # N x sent x 512 x 300
        sp_state = sp_state.max(1)[0]
        sp_logits = self.sp_linear(sp_state)
        type_state = torch.max(input_state, dim=1)[0]
        type_logits = self.type_linear(type_state)

        # 找结束位置用的开始和结束位置概率之和
        # (batch, 512, 1) + (batch, 1, 512) -> (512, 512)
        outer = start_logits[:, :, None] + end_logits[:, None]
        outer_mask = self.get_output_mask(outer)
        outer = outer - 1e30 * (1 - outer_mask[None].expand_as(outer))
        if query_mapping is not None:   # 这个是query_mapping (batch, 512)
            outer = outer - 1e30 * query_mapping[:, :, None]    # 不允许预测query的内容

        # 这两句相当于找到了outer中最大值的i和j坐标
        start_position = outer.max(dim=2)[0].max(dim=1)[1]
        end_position = outer.max(dim=1)[0].max(dim=1)[1]
        return start_logits, end_logits, type_logits, sp_logits.squeeze(2), start_position, end_position


from resnet import ResNet,BasicBlock,Bottleneck, resnet18,resnet34,resnet50,resnet101,resnet152, resnext50_32x4d, resnext101_32x8d, wide_resnet50_2, wide_resnet101_2
resnet_pool = dict(zip(range(9),[resnet18,resnet34,resnet50,resnet101,resnet152, resnext50_32x4d, resnext101_32x8d, wide_resnet50_2, wide_resnet101_2]))

class DeepCNNPredictionLayer(nn.Module):
    def __init__(self, config):
        super(DeepCNNPredictionLayer, self).__init__()
        self.input_dim = config.input_dim

        self.fc_hidden_size = config.fc_hidden_size
        self.dropout_size = config.dropout

        self.resnet = ResNet(block=BasicBlock, layers=[1, 1, 1, 1], num_classes=self.fc_hidden_size)
        self.dropout = nn.Dropout(self.dropout_size)

        self.sp_linear = nn.Linear(self.fc_hidden_size, 1)
        self.start_linear = nn.Linear(self.fc_hidden_size, 1)
        self.end_linear = nn.Linear(self.fc_hidden_size, 1)
        self.type_linear = nn.Linear(self.fc_hidden_size, config.label_type_num)  # yes/no/ans/unknown
        self.cache_S = 0
        self.cache_mask = None

    def get_output_mask(self, outer):
        # (batch, 512, 512)
        S = outer.size(1)
        if S <= self.cache_S:
            return Variable(self.cache_mask[:S, :S], requires_grad=False)
        self.cache_S = S
        # triu 生成上三角矩阵，tril生成下三角矩阵，这个相当于生成了(512, 512)的矩阵表示开始-结束的位置，答案长度最长为15
        np_mask = np.tril(np.triu(np.ones((S, S)), 0), 15)
        self.cache_mask = outer.data.new(S, S).copy_(torch.from_numpy(np_mask))
        return Variable(self.cache_mask, requires_grad=False)

    def forward(self, batch, input_state):
        query_mapping = batch['query_mapping']  # (batch, 512) 不一定是512，可能略小
        context_mask = batch['context_mask']  # bert里实际有输入的位置
        all_mapping = batch['all_mapping']  # (batch_size, 512, max_sent) 每个句子的token对应为1

        x = input_state.transpose(1, 2)#.type(torch.cuda.FloatTensor)
        x = self.resnet(x)
        input_state = x.transpose(2, 1)#.type(torch.cuda.FloatTensor)

        #input_state, support_state, type_state = self.dropout(cnn), self.dropout(cnn), self.dropout(cnn)

        start_logits = self.start_linear(input_state).squeeze(2) - 1e30 * (1 - context_mask)
        end_logits = self.end_linear(input_state).squeeze(2) - 1e30 * (1 - context_mask)
        sp_state = all_mapping.unsqueeze(3) * input_state.unsqueeze(2)  # N x sent x 512 x 300
        sp_state = sp_state.max(1)[0]
        sp_logits = self.sp_linear(sp_state)
        type_state = torch.max(input_state, dim=1)[0]
        type_logits = self.type_linear(type_state)

        # 找结束位置用的开始和结束位置概率之和
        # (batch, 512, 1) + (batch, 1, 512) -> (512, 512)
        outer = start_logits[:, :, None] + end_logits[:, None]
        outer_mask = self.get_output_mask(outer)
        outer = outer - 1e30 * (1 - outer_mask[None].expand_as(outer))
        if query_mapping is not None:  # 这个是query_mapping (batch, 512)
            outer = outer - 1e30 * query_mapping[:, :, None]  # 不允许预测query的内容

        # 这两句相当于找到了outer中最大值的i和j坐标
        start_position = outer.max(dim=2)[0].max(dim=1)[1]
        end_position = outer.max(dim=1)[0].max(dim=1)[1]
        return start_logits, end_logits, type_logits, sp_logits.squeeze(2), start_position, end_position
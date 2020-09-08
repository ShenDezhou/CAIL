import torch
import torch.nn as nn
import torch.nn.functional as F

from model.encoder.LSTMEncoder import LSTMEncoder
from model.layer.Attention import Attention
from tools.accuracy_tool import single_label_top1_accuracy, multi_label_top1_accuracy
from model.qa.util import generate_ans, multi_generate_ans


class Model(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(Model, self).__init__()

        self.hidden_size = config.getint("model", "hidden_size")
        self.word_num = 0
        f = open(config.get("data", "word2id"), "r", encoding="utf8")
        for line in f:
            self.word_num += 1

        self.embedding = nn.Embedding(self.word_num, self.hidden_size)
        self.context_encoder = LSTMEncoder(config, gpu_list, *args, **params)
        self.question_encoder = LSTMEncoder(config, gpu_list, *args, **params)
        self.attention = Attention(config, gpu_list, *args, **params)

        self.rank_module = nn.Linear(self.hidden_size * 2, 1)

        self.criterion = nn.CrossEntropyLoss()

        self.multi_module = nn.Linear(4, 16)
        self.accuracy_function = single_label_top1_accuracy

    def init_multi_gpu(self, device, config, *args, **params):
        pass

    def forward(self, data, config, gpu_list, acc_result, mode):
        context = data["context"]
        question = data["question"]

        batch = question.size()[0]
        option = question.size()[1]

        context = context.view(batch * option, -1)
        question = question.view(batch * option, -1)
        context = self.embedding(context)
        question = self.embedding(question)

        _, context = self.context_encoder(context)
        _, question = self.question_encoder(question)

        c, q, a = self.attention(context, question)

        y = torch.cat([torch.max(c, dim=1)[0], torch.max(q, dim=1)[0]], dim=1)

        y = y.view(batch * option, -1)
        y = self.rank_module(y)

        y = y.view(batch, option)

        y = self.multi_module(y)

        if mode != "test":
            label = data["label"]
            loss = self.criterion(y, label)
            acc_result = self.accuracy_function(y, label, config, acc_result)
            return {"loss": loss, "acc_result": acc_result}

        return {"output": generate_ans(data["id"], y)}


class ModelS(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(ModelS, self).__init__()

        self.hidden_size = config.getint("model", "hidden_size")
        self.word_num = 0
        f = open(config.get("data", "word2id"), "r", encoding="utf8")
        for line in f:
            self.word_num += 1

        self.context_len = config.getint("data", "max_option_len") * 4
        self.question_len = config.getint("data", "max_question_len")

        self.embedding = nn.Embedding(self.word_num, self.hidden_size)
        self.context_encoder = BertEncoder(config, gpu_list, *args, **params)
        self.question_encoder = BertEncoder(config, gpu_list, *args, **params)
        self.attention = Attention(config, gpu_list, *args, **params)
        self.dropout = nn.Dropout(config.getfloat("model", "dropout"))

        self.bce = nn.MultiLabelSoftMarginLoss(reduction='sum')
        self.gelu = nn.GELU()
        # self.fc_module_q = nn.Linear(self.question_len, 1)
        self.fc_module = nn.Linear(self.hidden_size * 2, 4)
        self.accuracy_function = multi_label_top1_accuracy

    def init_multi_gpu(self, device, config, *args, **params):
        pass

    def forward(self, data, config, gpu_list, acc_result, mode):
        context = data["context"]
        question = data["question"]

        batch = question.size()[0]
        option = question.size()[1]

        context = context.view(batch * option, -1)
        question = question.view(batch * option, -1)
        context = self.embedding(context)
        question = self.embedding(question)

        _, context = self.context_encoder(context)
        _, question = self.question_encoder(question)

        c, q, a = self.attention(context, question)

        # y = torch.cat([torch.max(c, dim=1)[0], torch.max(q, dim=1)[0]], dim=1)
        y = torch.cat([torch.mean(c, dim=1), torch.mean(q, dim=1)], dim=1)

        # y = y.view(batch * option, -1)
        # y = self.rank_module(y)
        # y = self.fc_module_q(a).squeeze(dim=2)
        y = self.gelu(y)
        y = self.dropout(y)
        # y = y.view(batch, option)
        y = self.fc_module(y)
        # y = torch.sigmoid(y)

        if mode != "test":
            label = data["label"]
            loss = self.bce(y, label)
            acc_result = self.accuracy_function(y, label, config, acc_result)
            return {"loss": loss, "acc_result": acc_result}

        return {"output": multi_generate_ans(data["id"], y)}


from model.encoder.BertEncoder import BertEncoder
class ModelX(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(ModelX, self).__init__()

        self.hidden_size = config.getint("model", "hidden_size")
        self.word_num = 0
        f = open(config.get("data", "word2id"), "r", encoding="utf8")
        for line in f:
            self.word_num += 1

        self.context_len = config.getint("data", "max_option_len") * 4
        self.question_len = config.getint("data", "max_question_len")

        self.embedding = nn.Embedding(self.word_num, self.hidden_size)
        self.context_encoder = BertEncoder(config, gpu_list, *args, **params)
        self.question_encoder = BertEncoder(config, gpu_list, *args, **params)
        self.attention = Attention(config, gpu_list, *args, **params)
        self.dropout = nn.Dropout(config.getfloat("model", "dropout"))

        self.bce = nn.MultiLabelSoftMarginLoss(reduction='sum')
        self.gelu = nn.GELU()
        # self.fc_module_q = nn.Linear(self.question_len, 1)
        self.fc_module = nn.Linear(self.hidden_size * 2, 4)
        self.accuracy_function = multi_label_top1_accuracy

    def init_multi_gpu(self, device, config, *args, **params):
        pass

    def forward(self, data, config, gpu_list, acc_result, mode):
        context = data["context"]
        question = data["question"]

        batch = question[0].size()[0]
        _, _, context = self.context_encoder(*context)
        _, _, question = self.question_encoder(*question)

        context_1 = context[-1].view(batch, -1, self.hidden_size)
        question_1 = question[-1]

        context_2 = context[-2].view(batch, -1, self.hidden_size)
        question_2 = question[-2]

        context = torch.cat([context_1,context_2], dim=1)
        question = torch.cat([question_1, question_2], dim=1)

        # c, q, a = self.attention(context, question)
        c, q = context, question
        # y = torch.cat([torch.max(c, dim=1)[0], torch.max(q, dim=1)[0]], dim=1)
        y = torch.cat([torch.mean(c, dim=1), torch.mean(q, dim=1)], dim=1)

        y = self.gelu(y)
        y = self.dropout(y)
        y = self.fc_module(y)


        if mode != "test":
            label = data["label"]
            loss = self.bce(y, label)
            acc_result = self.accuracy_function(y, label, config, acc_result)
            return {"loss": loss, "acc_result": acc_result}

        return {"output": multi_generate_ans(data["id"], y)}


from model.encoder.GRUEncoder import GRUEncoder
from model.qa.resnet import ResNet, BasicBlock
class RESModel(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(RESModel, self).__init__()

        self.hidden_size = config.getint("model", "hidden_size")
        self.word_num = 0
        f = open(config.get("data", "word2id"), "r", encoding="utf8")
        for line in f:
            self.word_num += 1

        self.embedding = nn.Embedding(self.word_num, self.hidden_size)
        self.context_encoder = GRUEncoder(config, gpu_list, *args, **params)
        self.question_encoder = GRUEncoder(config, gpu_list, *args, **params)
        self.attention = Attention(config, gpu_list, *args, **params)

        hidden_size = config.getint("model", "output_channel")  # config.num_fc_hidden_size
        self.resnet = ResNet(block=BasicBlock, layers=[0, 0, 0, 0], num_classes=hidden_size)
        self.rank_module = nn.Linear(hidden_size, 1)

        self.criterion = nn.CrossEntropyLoss()
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')

        self.multi_module = nn.Linear(4, 16)
        self.accuracy_function = single_label_top1_accuracy

    def init_multi_gpu(self, device, config, *args, **params):
        pass

    def forward(self, data, config, gpu_list, acc_result, mode):
        context = data["context"]
        question = data["question"]

        batch = question.size()[0]
        option = question.size()[1]

        context = context.view(batch * option, -1)
        question = question.view(batch * option, -1)
        context = self.embedding(context)
        question = self.embedding(question)

        _, context = self.context_encoder(context)
        _, question = self.question_encoder(question)

        c, q, a = self.attention(context, question)

        # x = torch.cat([torch.max(c, dim=1)[0], torch.max(q, dim=1)[0]], dim=1)
        x = torch.cat([c, q], dim=1)
        if torch.cuda.is_available():
            x = x.transpose(1, 2).type(torch.cuda.FloatTensor)
        else:
            x = x.transpose(1, 2).type(torch.FloatTensor)

        x = self.resnet(x)
        y = x.view(batch * option, -1)
        y = self.rank_module(y)
        y = y.view(batch, option)
        y = self.multi_module(y)


        if mode != "test":
            label = data["label"]
            loss = self.criterion(y, label)
            acc_result = self.accuracy_function(y, label, config, acc_result)
            return {"loss": loss, "acc_result": acc_result}

        return {"output": generate_ans(data["id"], y)}


from model.qa.capsnetx import PrimaryCaps, FCCaps, FlattenCaps
class CAPSModel(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(CAPSModel, self).__init__()

        self.hidden_size = config.getint("model", "hidden_size")
        self.word_num = 0
        f = open(config.get("data", "word2id"), "r", encoding="utf8")
        for line in f:
            self.word_num += 1

        self.embedding = nn.Embedding(self.word_num, self.hidden_size)
        self.context_encoder = LSTMEncoder(config, gpu_list, *args, **params)
        self.question_encoder = LSTMEncoder(config, gpu_list, *args, **params)
        self.attention = Attention(config, gpu_list, *args, **params)

        self.num_classes = 4
        # self.conv_channel = config.getint("data", "max_question_len") + config.getint("data", "max_option_len")

        self.dim_capsule = config.getint("model", "dim_capsule")
        self.num_compressed_capsule = config.getint("model", "num_compressed_capsule")
        self.ngram_size = [2, 4, 8]
        self.convs_doc = nn.ModuleList([nn.Conv1d(self.hidden_size, 32, K, stride=2) for K in self.ngram_size])
        torch.nn.init.xavier_uniform_(self.convs_doc[0].weight)
        torch.nn.init.xavier_uniform_(self.convs_doc[1].weight)
        torch.nn.init.xavier_uniform_(self.convs_doc[2].weight)

        self.primary_capsules_doc = PrimaryCaps(num_capsules=self.dim_capsule, in_channels=32, out_channels=32,
                                                kernel_size=1, stride=1)

        self.flatten_capsules = FlattenCaps()

        self.W_doc = nn.Parameter(torch.FloatTensor(49024, self.num_compressed_capsule))
        torch.nn.init.xavier_uniform_(self.W_doc)

        self.fc_capsules_doc_child = FCCaps(config, output_capsule_num=self.num_classes,
                                            input_capsule_num=self.num_compressed_capsule,
                                            in_channels=self.dim_capsule, out_channels=self.dim_capsule)

        # self.rank_module = nn.Linear(hidden_size, 1)

        # self.criterion = nn.CrossEntropyLoss()
        self.bce = nn.BCELoss(reduction='mean')

        self.fc_module = nn.Linear(self.dim_capsule, self.num_classes)
        self.accuracy_function = multi_label_top1_accuracy

    def compression(self, poses, W):
        poses = torch.matmul(poses.permute(0, 2, 1), W).permute(0, 2, 1)
        activations = torch.sqrt((poses ** 2).sum(2))
        return poses, activations

    def init_multi_gpu(self, device, config, *args, **params):
        pass

    def forward(self, data, config, gpu_list, acc_result, mode):
        context = data["context"]
        question = data["question"]

        batch = question.size()[0]
        option = question.size()[1]

        context = context.view(batch, -1)
        question = question.view(batch, -1)
        context = self.embedding(context)
        question = self.embedding(question)

        _, context = self.context_encoder(context)
        _, question = self.question_encoder(question)

        c, q, a = self.attention(context, question)

        # x = torch.cat([torch.max(c, dim=1)[0], torch.max(q, dim=1)[0]], dim=1)
        x = torch.cat([c, q], dim=1)
        if torch.cuda.is_available():
            x = x.transpose(1, 2).type(torch.cuda.FloatTensor)
        else:
            x = x.transpose(1, 2).type(torch.FloatTensor)

        nets_doc_l = []
        for i in range(len(self.ngram_size)):
            nets = self.convs_doc[i](x)
            nets_doc_l.append(nets)
        nets_doc = torch.cat((nets_doc_l[0], nets_doc_l[1], nets_doc_l[2]), 2)
        poses_doc, activations_doc = self.primary_capsules_doc(nets_doc)
        poses, activations = self.flatten_capsules(poses_doc, activations_doc)
        poses, activations = self.compression(poses, self.W_doc)
        poses, type_logits = self.fc_capsules_doc_child(poses, activations, range(4))  # 4 types in total.
        type_logits = type_logits.squeeze(2)
        # type_logits = self.fc_module(type_logits)

        # y = x.view(batch * option, -1)
        # y = self.rank_module(y)
        # y = y.view(batch, option)
        # y = self.multi_module(y)

        if mode != "test":
            label = data["label"]
            # loss = self.criterion(type_logits, label)
            loss = self.bce(type_logits, label)
            acc_result = self.accuracy_function(type_logits, label, config, acc_result)
            return {"loss": loss, "acc_result": acc_result}

        return {"output": multi_generate_ans(data["id"], type_logits)}


class BertCAPSModel(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(BertCAPSModel, self).__init__()

        self.hidden_size = config.getint("model", "hidden_size")
        self.word_num = 0
        f = open(config.get("data", "word2id"), "r", encoding="utf8")
        for line in f:
            self.word_num += 1

        self.embedding = nn.Embedding(self.word_num, self.hidden_size)
        self.context_encoder = BertEncoder(config, gpu_list, *args, **params)
        self.question_encoder = LSTMEncoder(config, gpu_list, *args, **params)
        self.attention = Attention(config, gpu_list, *args, **params)

        self.num_classes = 4
        # self.conv_channel = config.getint("data", "max_question_len") + config.getint("data", "max_option_len")
        self.question_len = config.getint("data", "max_question_len")
        self.context_len = config.getint("data", "max_option_len") * 4
        self.dim_capsule = config.getint("model", "dim_capsule")
        self.num_compressed_capsule = config.getint("model", "num_compressed_capsule")
        self.ngram_size = [2, 4, 8]
        self.convs_doc = nn.ModuleList([nn.Conv1d(self.hidden_size, 32, K, stride=2) for K in self.ngram_size])
        torch.nn.init.xavier_uniform_(self.convs_doc[0].weight)
        torch.nn.init.xavier_uniform_(self.convs_doc[1].weight)
        torch.nn.init.xavier_uniform_(self.convs_doc[2].weight)

        self.primary_capsules_doc = PrimaryCaps(num_capsules=self.dim_capsule, in_channels=32, out_channels=32,
                                                kernel_size=1, stride=1)

        self.flatten_capsules = FlattenCaps()

        self.W_doc = nn.Parameter(torch.FloatTensor(49024, self.num_compressed_capsule))
        torch.nn.init.xavier_uniform_(self.W_doc)

        self.fc_capsules_doc_child = FCCaps(config, output_capsule_num=self.num_classes,
                                            input_capsule_num=self.num_compressed_capsule,
                                            in_channels=self.dim_capsule, out_channels=self.dim_capsule)

        # self.rank_module = nn.Linear(hidden_size, 1)

        self.criterion = nn.CrossEntropyLoss()
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')

        self.fc_module = nn.Linear(self.dim_capsule, self.num_classes)
        self.accuracy_function = multi_label_top1_accuracy

    def compression(self, poses, W):
        poses = torch.matmul(poses.permute(0, 2, 1), W).permute(0, 2, 1)
        activations = torch.sqrt((poses ** 2).sum(2))
        return poses, activations

    def init_multi_gpu(self, device, config, *args, **params):
        pass

    def forward(self, data, config, gpu_list, acc_result, mode):
        context = data["context"]
        question = data["question"]

        batch = question.size()[0]
        option = question.size()[1]

        # context = context.view(batch, -1)
        question = question.view(batch, -1)
        # context = self.embedding(context)
        question = self.embedding(question)

        context = self.context_encoder(context, self.context_len)
        _, question = self.question_encoder(question)

        c, q, a = self.attention(context[-1], question)

        # x = torch.cat([torch.max(c, dim=1)[0], torch.max(q, dim=1)[0]], dim=1)
        x = torch.cat([c, q], dim=1)
        if torch.cuda.is_available():
            x = x.transpose(1, 2).type(torch.cuda.FloatTensor)
        else:
            x = x.transpose(1, 2).type(torch.FloatTensor)

        nets_doc_l = []
        for i in range(len(self.ngram_size)):
            nets = self.convs_doc[i](x)
            nets_doc_l.append(nets)
        nets_doc = torch.cat((nets_doc_l[0], nets_doc_l[1], nets_doc_l[2]), 2)
        poses_doc, activations_doc = self.primary_capsules_doc(nets_doc)
        poses, activations = self.flatten_capsules(poses_doc, activations_doc)
        poses, activations = self.compression(poses, self.W_doc)
        poses, type_logits = self.fc_capsules_doc_child(poses, activations, range(4))  # 4 types in total.
        type_logits = type_logits.squeeze(2)
        # type_logits = self.fc_module(type_logits)

        # y = x.view(batch * option, -1)
        # y = self.rank_module(y)
        # y = y.view(batch, option)
        # y = self.multi_module(y)

        if mode != "test":
            label = data["label"]
            # loss = self.criterion(type_logits, label)
            loss = self.bce(type_logits, label)
            acc_result = self.accuracy_function(type_logits, label, config, acc_result)
            return {"loss": loss, "acc_result": acc_result}

        return {"output": multi_generate_ans(data["id"], type_logits)}

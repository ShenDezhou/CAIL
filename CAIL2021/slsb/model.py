"""BERT and RNN model for sentence pair classification.

Author: Tsinghuaboy (tsinghua9boy@sina.com)

Used for SMP-CAIL2020-Argmine.
"""
import copy

import torch

from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence
from transformers import BertModel
from transformers import AutoModel
import torch.nn.functional as F

class BertForClassification(nn.Module):
    """BERT with simple linear model."""
    def __init__(self, config):
        """Initialize the model with config dict.

        Args:
            config: python dict must contains the attributes below:
                config.bert_model_path: pretrained model path or model type
                    e.g. 'bert-base-chinese'
                config.hidden_size: The same as BERT model, usually 768
                config.num_classes: int, e.g. 2
                config.dropout: float between 0 and 1
        """
        super().__init__()
        self.bert = BertModel.from_pretrained(config.bert_model_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.dropout = nn.Dropout(config.dropout)
        self.linear = nn.Linear(4, config.num_classes)
        self.num_classes = config.num_classes

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

        if config.hidden_size == 768:
            self.W_doc = nn.Parameter(torch.FloatTensor(147328, self.num_compressed_capsule))
        else:#1024
            self.W_doc = nn.Parameter(torch.FloatTensor(196480, self.num_compressed_capsule))
        torch.nn.init.xavier_uniform_(self.W_doc)

        self.fc_capsules_doc_child = FCCaps(config, output_capsule_num=config.num_classes,
                                            input_capsule_num=self.num_compressed_capsule,
                                            in_channels=self.dim_capsule, out_channels=self.dim_capsule)


    def compression(self, poses, W):
        poses = torch.matmul(poses.permute(0, 2, 1), W).permute(0, 2, 1)
        activations = torch.sqrt((poses ** 2).sum(2))
        return poses, activations



    def forward(self, input_ids, attention_mask, token_type_ids):
        """Forward inputs and get logits.

        Args:
            input_ids: (batch_size, max_seq_len)
            attention_mask: (batch_size, max_seq_len)
            token_type_ids: (batch_size, max_seq_len)

        Returns:
            logits: (batch_size, num_classes)
        """
        batch_size = input_ids.shape[0]
        hiddens = self.bert(input_ids=input_ids, attention_mask=attention_mask,token_type_ids=token_type_ids,
                                                                    output_hidden_states=True)[2]
        hidden_state = torch.cat([*hiddens[-3:], hiddens[0]], dim=2)
        # bert_output[0]: (batch_size, sequence_length, hidden_size)
        # bert_output[1]: (batch_size, hidden_size)
        #hidden_state = hidden_state.mean(1)
        #hidden_state = self.dropout(hidden_state)
        #logits = self.linear(hidden_state).view(batch_size, self.num_classes)
        #logits = torch.sigmoid(logits)
        # logits: (batch_size, num_classes)
        nets_doc_l = []
        for i in range(len(self.ngram_size)):
            nets = self.convs_doc[i](hidden_state)
            nets_doc_l.append(nets)
        nets_doc = torch.cat((nets_doc_l[0], nets_doc_l[1], nets_doc_l[2]), 2)
        poses_doc, activations_doc = self.primary_capsules_doc(nets_doc)
        poses, activations = self.flatten_capsules(poses_doc, activations_doc)
        poses, activations = self.compression(poses, self.W_doc)
        poses, logits = self.fc_capsules_doc_child(poses, activations, range(4))#4 types in total.

        logits = self.linear(logits.view(batch_size,-1)).view(batch_size, self.num_classes)
        return logits

class BertXForClassification(nn.Module):
    """BERT with simple linear model."""
    def __init__(self, config):
        """Initialize the model with config dict.

        Args:
            config: python dict must contains the attributes below:
                config.bert_model_path: pretrained model path or model type
                    e.g. 'bert-base-chinese'
                config.hidden_size: The same as BERT model, usually 768
                config.num_classes: int, e.g. 2
                config.dropout: float between 0 and 1
        """
        super().__init__()
        self.bert = BertModel.from_pretrained(config.bert_model_path)
        for param in self.bert.parameters():
            param.requires_grad = True

        # data(b, 512, 768) -> conv(b, 511,767) -> bn -> mp(b, 4, 6)
        self.conv_module = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(128, 128), stride=(128, 128), padding=(1, 1))
        )
        # data(b, 512, 768) -> conv(b, 255,255) -> bn -> mp(b, 4, 4)
        self.conv_module2 = nn.Sequential(
            nn.Conv2d(1,1, kernel_size=(2,3), stride=(2,3),padding=(0,0)),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(64, 64), stride=(64, 64),padding=(1,1))
        )
        # data(b, 512, 768) -> conv(b, 169, 192) -> bn -> mp(b, 5, 6)
        self.conv_module3 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(3, 4), stride=(3, 4), padding=(0, 0)),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(32, 32), stride=(32, 32), padding=(1, 1))
        )
        # data(b, 512, 768) -> conv(b, 127, 127) -> bn -> mp(b, 4, 4)
        self.conv_module4 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(4, 6), stride=(4, 6), padding=(0, 0)),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(32, 32), stride=(32, 32), padding=(1, 1))
        )
        # data(b, 512, 768) -> conv(b, 101, 108) -> bn -> mp(b, 6, 6)
        self.conv_module5 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(5, 7), stride=(5, 7), padding=(0, 0)),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(16, 16), stride=(16, 16), padding=(1, 1))
        )
        # data(b, 512, 768) -> conv(b, 84, 84) -> bn -> mp(b, 5, 5)
        self.conv_module6 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(6, 9), stride=(6, 9), padding=(0, 0)),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(16, 16), stride=(16, 16), padding=(1, 1))
        )
        # data(b, 512, 768) -> conv(b, 72, 75) -> bn -> mp(b, 9, 9)
        self.conv_module7 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(7, 10), stride=(7, 10), padding=(0, 0)),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(8, 8), stride=(8, 8), padding=(1, 1))
        )
        #cnn feature map has a total number of 228 dimensions.
        self.dropout = nn.Dropout(config.dropout)
        self.linear = nn.Linear(config.hidden_size+228, config.num_classes)
        self.bn = nn.BatchNorm1d(config.num_classes)
        self.num_classes = config.num_classes

    def forward(self, input_ids, attention_mask, token_type_ids):
        """Forward inputs and get logits.

        Args:
            input_ids: (batch_size, max_seq_len)
            attention_mask: (batch_size, max_seq_len)
            token_type_ids: (batch_size, max_seq_len)

        Returns:
            logits: (batch_size, num_classes)
        """
        batch_size = input_ids.shape[0]
        bert_output = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            # encoder_hidden_states=False
        )
        # bert_output[0]: (batch_size, sequence_length, hidden_size)
        encoded_output = bert_output[0]
        # encoded_output[0]: (batch_size, 1, sequence_length, hidden_size)
        encoded_output = encoded_output.view(batch_size, 1, encoded_output.shape[1], -1)
        cnn_feats = []
        cnn_feats.append(self.conv_module(encoded_output))
        cnn_feats.append(self.conv_module2(encoded_output))
        cnn_feats.append(self.conv_module3(encoded_output))
        cnn_feats.append(self.conv_module4(encoded_output))
        cnn_feats.append(self.conv_module5(encoded_output))
        cnn_feats.append(self.conv_module6(encoded_output))
        cnn_feats.append(self.conv_module7(encoded_output))
        for index in range(len(cnn_feats)):
            cnn_feats[index] = cnn_feats[index].reshape((batch_size, -1))
        con_cnn_feats = torch.cat(cnn_feats, dim=1)

        # bert_output[1]: (batch_size, hidden_size)
        pooled_output = bert_output[1]
        # 228 + 768 ->
        pooled_output = torch.cat([con_cnn_feats, pooled_output], dim=1)
        pooled_output = self.dropout(pooled_output)
        logits = self.linear(pooled_output).view(batch_size, self.num_classes)
        logits = self.bn(logits)
        logits = nn.functional.softmax(logits, dim=-1)
        # logits: (batch_size, num_classes)
        return logits

class BertYForClassification(nn.Module):
    """BERT with simple linear model."""
    def __init__(self, config):
        """Initialize the model with config dict.

        Args:
            config: python dict must contains the attributes below:
                config.bert_model_path: pretrained model path or model type
                    e.g. 'bert-base-chinese'
                config.hidden_size: The same as BERT model, usually 768
                config.num_classes: int, e.g. 2
                config.dropout: float between 0 and 1
        """
        super().__init__()
        self.bert = BertModel.from_pretrained(config.bert_model_path)
        for param in self.bert.parameters():
            param.requires_grad = True

        # data(b, 512, 768) -> conv(b, 511,767) -> bn -> mp(b, 4, 6)
        self.conv_module = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(128, 128), stride=(128, 128), padding=(1, 1))
        )
        # data(b, 512, 768) -> conv(b, 255,255) -> bn -> mp(b, 4, 4)
        self.conv_module2 = nn.Sequential(
            nn.Conv2d(1,1, kernel_size=(2,3), stride=(2,3),padding=(0,0)),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(64, 64), stride=(64, 64),padding=(1,1))
        )
        # data(b, 512, 768) -> conv(b, 169, 192) -> bn -> mp(b, 5, 6)
        self.conv_module3 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(3, 4), stride=(3, 4), padding=(0, 0)),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(32, 32), stride=(32, 32), padding=(1, 1))
        )
        # data(b, 512, 768) -> conv(b, 127, 127) -> bn -> mp(b, 4, 4)
        self.conv_module4 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(4, 6), stride=(4, 6), padding=(0, 0)),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(32, 32), stride=(32, 32), padding=(1, 1))
        )
        # data(b, 512, 768) -> conv(b, 101, 108) -> bn -> mp(b, 6, 6)
        self.conv_module5 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(5, 7), stride=(5, 7), padding=(0, 0)),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(16, 16), stride=(16, 16), padding=(1, 1))
        )
        # data(b, 512, 768) -> conv(b, 84, 84) -> bn -> mp(b, 5, 5)
        self.conv_module6 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(6, 9), stride=(6, 9), padding=(0, 0)),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(16, 16), stride=(16, 16), padding=(1, 1))
        )
        # data(b, 512, 768) -> conv(b, 72, 75) -> bn -> mp(b, 9, 9)
        self.conv_module7 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(7, 10), stride=(7, 10), padding=(0, 0)),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(8, 8), stride=(8, 8), padding=(1, 1))
        )
        self.conv_module8 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(8, 12), stride=(8, 12), padding=(0, 0)),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(8, 8), stride=(8, 8), padding=(1, 1))
        )
        # data(b, 512, 768) -> conv(b, 255,255) -> bn -> mp(b, 4, 4)
        self.conv_module9 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(9, 13), stride=(9, 13), padding=(0, 0)),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(8, 8), stride=(8, 8), padding=(1, 1))
        )
        # data(b, 512, 768) -> conv(b, 169, 192) -> bn -> mp(b, 5, 6)
        self.conv_moduleA = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(10, 15), stride=(10, 15), padding=(0, 0)),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4, 4), stride=(4, 4), padding=(1, 1))
        )
        # data(b, 512, 768) -> conv(b, 127, 127) -> bn -> mp(b, 4, 4)
        self.conv_moduleB = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(11, 16), stride=(11, 16), padding=(0, 0)),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4, 4), stride=(4, 4), padding=(1, 1))
        )
        # data(b, 512, 768) -> conv(b, 101, 108) -> bn -> mp(b, 6, 6)
        self.conv_moduleC = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(12, 18), stride=(12, 18), padding=(0, 0)),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(1, 1))
        )
        # data(b, 512, 768) -> conv(b, 84, 84) -> bn -> mp(b, 5, 5)
        self.conv_moduleD = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(13, 19), stride=(13, 19), padding=(0, 0)),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(1, 1))
        )
        # data(b, 512, 768) -> conv(b, 72, 75) -> bn -> mp(b, 9, 9)
        self.conv_moduleE = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(14, 21), stride=(14, 21), padding=(0, 0)),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(1, 1))
        )

        #cnn feature map has a total number of 228 dimensions.
        self.dropout = nn.Dropout(config.dropout)
        # 1-7: 228; 8-14: 1691
        self.linear = nn.Linear(config.hidden_size + 1005, config.num_classes)
        #self.bn = nn.BatchNorm1d(config.num_classes)
        self.num_classes = config.num_classes

    def forward(self, input_ids, attention_mask, token_type_ids):
        """Forward inputs and get logits.

        Args:
            input_ids: (batch_size, max_seq_len)
            attention_mask: (batch_size, max_seq_len)
            token_type_ids: (batch_size, max_seq_len)

        Returns:
            logits: (batch_size, num_classes)
        """
        batch_size = input_ids.shape[0]
        bert_output = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            # encoder_hidden_states=False
        )
        # bert_output[0]: (batch_size, sequence_length, hidden_size)
        encoded_output = bert_output[0]
        # encoded_output[0]: (batch_size, 1, sequence_length, hidden_size)
        encoded_output = encoded_output.view(batch_size, 1, encoded_output.shape[1], -1)
        cnn_feats = []
        cnn_feats.append(self.conv_module(encoded_output))
        cnn_feats.append(self.conv_module2(encoded_output))
        cnn_feats.append(self.conv_module3(encoded_output))
        cnn_feats.append(self.conv_module4(encoded_output))
        cnn_feats.append(self.conv_module5(encoded_output))
        cnn_feats.append(self.conv_module6(encoded_output))
        cnn_feats.append(self.conv_module7(encoded_output))
        cnn_feats.append(self.conv_module8(encoded_output))
        cnn_feats.append(self.conv_module9(encoded_output))
        cnn_feats.append(self.conv_moduleA(encoded_output))
        cnn_feats.append(self.conv_moduleB(encoded_output))
        cnn_feats.append(self.conv_moduleC(encoded_output))
        cnn_feats.append(self.conv_moduleD(encoded_output))
        cnn_feats.append(self.conv_moduleE(encoded_output))
        for index in range(len(cnn_feats)):
            cnn_feats[index] = cnn_feats[index].reshape((batch_size, -1))
        con_cnn_feats = torch.cat(cnn_feats, dim=1)

        # bert_output[1]: (batch_size, hidden_size)
        pooled_output = bert_output[1]
        # 228 + 768 ->
        pooled_output = torch.cat([con_cnn_feats, pooled_output], dim=1)
        pooled_output = self.dropout(pooled_output)
        logits = self.linear(pooled_output).view(batch_size, self.num_classes)
        #logits = self.bn(logits)
        logits = nn.functional.softmax(logits, dim=-1)
        # logits: (batch_size, num_classes)
        return logits

#A Hierarchical Multi-grained Transformer-based Document Summarization Method
class BertXLForClassification(nn.Module):
    """BERT with simple linear model."""
    def __init__(self, config):
        """Initialize the model with config dict.

        Args:
            config: python dict must contains the attributes below:
                config.bert_model_path: pretrained model path or model type
                    e.g. 'bert-base-chinese'
                config.hidden_size: The same as BERT model, usually 768
                config.num_classes: int, e.g. 2
                config.dropout: float between 0 and 1
        """
        super().__init__()
        if 'xl' in config.model_type:
            self.bert = AutoModel.from_pretrained(config.bert_model_path)
        else:
            self.bert = BertModel.from_pretrained(config.bert_model_path)

        for param in self.bert.parameters():
            param.requires_grad = True

        #cnn feature map has a total number of 228 dimensions.
        self.dropout = nn.Dropout(config.dropout)
        # 1-7: 228; 8-14: 1691
        self.linear = nn.Linear(config.hidden_size, config.num_classes)
        # self.linear_last = nn.Linear(config.max_seq_len, config.num_classes)
        #self.bn = nn.BatchNorm1d(config.num_classes)
        self.num_classes = config.num_classes

    def forward(self, input_ids, attention_mask, token_type_ids):
        """Forward inputs and get logits.

        Args:
            input_ids: (batch_size, max_seq_len)
            attention_mask: (batch_size, max_seq_len)
            token_type_ids: (batch_size, max_seq_len)

        Returns:
            logits: (batch_size, num_classes)
        """
        batch_size = input_ids.shape[0]
        bert_output = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            # encoder_hidden_states=False
        )

        encoded_output = bert_output[0]


        encoded_output = torch.mean(encoded_output, dim=1)
        pooled_output = self.dropout(encoded_output)

        # logits = logits.squeeze(dim=2)
        logits = self.linear(pooled_output)
        #logits = self.bn(logits)
        logits = nn.functional.softmax(logits, dim=-1)
        # logits: (batch_size, num_classes)
        return logits


class BertXLCForClassification(nn.Module):
    """BERT with simple linear model."""
    def __init__(self, config):
        """Initialize the model with config dict.

        Args:
            config: python dict must contains the attributes below:
                config.bert_model_path: pretrained model path or model type
                    e.g. 'bert-base-chinese'
                config.hidden_size: The same as BERT model, usually 768
                config.num_classes: int, e.g. 2
                config.dropout: float between 0 and 1
        """
        super().__init__()
        if 'xl' in config.model_type:
            self.bert = AutoModel.from_pretrained(config.bert_model_path)
        else:
            self.bert = BertModel.from_pretrained(config.bert_model_path)

        for param in self.bert.parameters():
            param.requires_grad = True
        self.dropout = nn.Dropout(config.dropout)
        self.linear = nn.Linear(4, config.num_classes)
        self.num_classes = config.num_classes

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

        if config.hidden_size == 768:
            self.W_doc = nn.Parameter(torch.FloatTensor(147328, self.num_compressed_capsule))
        else:#1024
            self.W_doc = nn.Parameter(torch.FloatTensor(196480, self.num_compressed_capsule))
        torch.nn.init.xavier_uniform_(self.W_doc)

        self.fc_capsules_doc_child = FCCaps(config, output_capsule_num=config.num_classes,
                                            input_capsule_num=self.num_compressed_capsule,
                                            in_channels=self.dim_capsule, out_channels=self.dim_capsule)


    def compression(self, poses, W):
        poses = torch.matmul(poses.permute(0, 2, 1), W).permute(0, 2, 1)
        activations = torch.sqrt((poses ** 2).sum(2))
        return poses, activations


    def forward(self, input_ids, attention_mask, token_type_ids):
        """Forward inputs and get logits.

        Args:
            input_ids: (batch_size, max_seq_len)
            attention_mask: (batch_size, max_seq_len)
            token_type_ids: (batch_size, max_seq_len)

        Returns:
            logits: (batch_size, num_classes)
        """
        batch_size = input_ids.shape[0]
        hiddens = self.bert(input_ids=input_ids, attention_mask=attention_mask,token_type_ids=token_type_ids,
                                                                    output_hidden_states=True)[1]
        hidden_state = torch.cat([*hiddens[-3:], hiddens[0]], dim=2)
        # bert_output[0]: (batch_size, sequence_length, hidden_size)
        # bert_output[1]: (batch_size, hidden_size)
        nets_doc_l = []
        for i in range(len(self.ngram_size)):
            nets = self.convs_doc[i](hidden_state)
            nets_doc_l.append(nets)
        nets_doc = torch.cat((nets_doc_l[0], nets_doc_l[1], nets_doc_l[2]), 2)
        poses_doc, activations_doc = self.primary_capsules_doc(nets_doc)
        poses, activations = self.flatten_capsules(poses_doc, activations_doc)
        poses, activations = self.compression(poses, self.W_doc)
        poses, logits = self.fc_capsules_doc_child(poses, activations, range(4))#4 types in total.

        logits = self.linear(logits.view(batch_size,-1)).view(batch_size, self.num_classes)
        return logits


class WordKVMN(nn.Module):
    def __init__(self, config):
        super(WordKVMN, self).__init__()
        self.temper = config.hidden_size ** 0.5
        self.word_embedding_a = nn.Embedding(config.vocab_size, config.hidden_size * 2)
        # self.word_embedding_c = nn.Embedding(10, config.hidden_size)
        self.max_seq_len = config.max_seq_len

    def forward(self, hidden_state, word_seq, label_value_matrix):
        embedding_a = self.word_embedding_a(word_seq)
        # embedding_c = self.word_embedding_c(label_value_matrix)
        embedding_a = embedding_a.permute(0, 2, 1)
        p = torch.matmul(hidden_state, embedding_a) #/ self.temper
        # tmp_word_mask_metrix = torch.clamp(word_mask_metrix, 0, 1)

        # exp_u = torch.exp(u)
        # delta_exp_u = exp_u
        # delta_exp_u = torch.mul(exp_u, tmp_word_mask_metrix)

        # sum_delta_exp_u = torch.stack([torch.sum(delta_exp_u, 2)] * delta_exp_u.shape[2], 2)

        # p = torch.div(delta_exp_u, sum_delta_exp_u + 1e-10)

        # embedding_c = embedding_c.permute(3, 0, 1, 2)
        # o = torch.mul(p, embedding_c)

        # o = o.permute(1, 2, 3, 0)
        # o = torch.sum(o, 2)

        #o = torch.add(p, hidden_state)

        return p


class CRF(nn.Module):
    """Conditional random field.

    This module implements a conditional random field [LMP01]_. The forward computation
    of this class computes the log likelihood of the given sequence of tags and
    emission score tensor. This class also has `~CRF.decode` method which finds
    the best tag sequence given an emission score tensor using `Viterbi algorithm`_.

    Args:
        num_tags: Number of tags.
        batch_first: Whether the first dimension corresponds to the size of a minibatch.

    Attributes:
        start_transitions (`~torch.nn.Parameter`): Start transition score tensor of size
            ``(num_tags,)``.
        end_transitions (`~torch.nn.Parameter`): End transition score tensor of size
            ``(num_tags,)``.
        transitions (`~torch.nn.Parameter`): Transition score tensor of size
            ``(num_tags, num_tags)``.
    """

    def __init__(self, num_tags, batch_first):
        if num_tags <= 0:
            raise ValueError('invalid number of tags: {}'.format(num_tags))
        super(CRF, self).__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize the transition parameters.

        The parameters will be initialized randomly from a uniform distribution
        between -0.1 and 0.1.
        """
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def __repr__(self):
        return 'num_tags={}'.format(self.num_tags)

    def forward(self, emissions, tags, mask, reduction):
        """Compute the conditional log likelihood of a sequence of tags given emission scores.

        Args:
            emissions (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
            tags (`~torch.LongTensor`): Sequence of tags tensor of size
                ``(seq_length, batch_size)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length)`` otherwise.
            mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.
            reduction: Specifies  the reduction to apply to the output:
                ``none|sum|mean|token_mean``. ``none``: no reduction will be applied.
                ``sum``: the output will be summed over batches. ``mean``: the output will be
                averaged over batches. ``token_mean``: the output will be averaged over tokens.

        Returns:
            `~torch.Tensor`: The log likelihood. This will have size ``(batch_size,)`` if
            reduction is ``none``, ``()`` otherwise.
        """
        self._validate(emissions, tags=tags, mask=mask)
        if reduction not in ('none', 'sum', 'mean', 'token_mean'):
            raise ValueError('invalid reduction: {}'.format(reduction))
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.uint8)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)

        # shape: (batch_size,)
        numerator = self._compute_score(emissions, tags, mask)
        # shape: (batch_size,)
        denominator = self._compute_normalizer(emissions, mask)
        # shape: (batch_size,)
        llh = numerator - denominator

        if reduction == 'none':
            return llh
        if reduction == 'sum':
            return llh.sum()
        if reduction == 'mean':
            return llh.mean()
        assert reduction == 'token_mean'
        return llh.sum() / mask.float().sum()

    def decode(self, emissions, mask=None):
        """Find the most likely tag sequence using Viterbi algorithm.

        Args:
            emissions (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
            mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.

        Returns:
            List of list containing the best tag sequence for each batch.
        """
        self._validate(emissions, tags=None, mask=mask)
        if mask is None:
            mask = emissions.new_ones(emissions.shape[:2], dtype=torch.uint8)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)

        return self._viterbi_decode(emissions, mask)

    def _validate(self, emissions, tags, mask):
        if emissions.dim() != 3:
            raise ValueError('emissions must have dimension of 3, got {}'.format(emissions.dim()))
        if emissions.size(2) != self.num_tags:
            raise ValueError(
                'expected last dimension of emissions is {}, got {}'.format(self.num_tags, emissions.size(2)))

        if tags is not None:
            if emissions.shape[:2] != tags.shape:
                raise ValueError(
                    'the first two dimensions of emissions and tags must match,got {} and {}'.format(
                        tuple(emissions.shape[:2]), tuple(tags.shape)))

        if mask is not None:
            if emissions.shape[:2] != mask.shape:
                raise ValueError(
                    'the first two dimensions of emissions and mask must match, got {} and {}'.format(
                        tuple(emissions.shape[:2]), tuple(mask.shape)))
            no_empty_seq = not self.batch_first and mask[0].all()
            no_empty_seq_bf = self.batch_first and mask[:, 0].all()
            if not no_empty_seq and not no_empty_seq_bf:
                raise ValueError('mask of the first timestep must all be on')

    def _compute_score(
            self, emissions, tags, mask):
        # emissions: (seq_length, batch_size, num_tags)
        # tags: (seq_length, batch_size)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and tags.dim() == 2
        assert emissions.shape[:2] == tags.shape
        assert emissions.size(2) == self.num_tags
        assert mask.shape == tags.shape
        assert mask[0].all()

        seq_length, batch_size = tags.shape
        mask = mask.float()

        # Start transition score and first emission
        # shape: (batch_size,)
        score = self.start_transitions[tags[0]]
        score += emissions[0, torch.arange(batch_size), tags[0]]

        for i in range(1, seq_length):
            # Transition score to next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            score += self.transitions[tags[i - 1], tags[i]] * mask[i]

            # Emission score for next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            score += emissions[i, torch.arange(batch_size), tags[i]] * mask[i]

        # End transition score
        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1
        # shape: (batch_size,)
        last_tags = tags[seq_ends, torch.arange(batch_size)]
        # shape: (batch_size,)
        score += self.end_transitions[last_tags]

        return score

    def _compute_normalizer(self, emissions, mask):
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        assert mask[0].all()

        seq_length = emissions.size(0)

        # Start transition score and first emission; score has size of
        # (batch_size, num_tags) where for each batch, the j-th column stores
        # the score that the first timestep has tag j
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0]

        for i in range(1, seq_length):
            # Broadcast score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emissions = emissions[i].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the sum of scores of all
            # possible tag sequences so far that end with transitioning from tag i to tag j
            # and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emissions

            # Sum over all possible current tags, but we're in score space, so a sum
            # becomes a log-sum-exp: for each sample, entry i stores the sum of scores of
            # all possible tag sequences so far, that end in tag i
            # shape: (batch_size, num_tags)
            next_score = torch.logsumexp(next_score, dim=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # shape: (batch_size, num_tags)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)

        # End transition score
        # shape: (batch_size, num_tags)
        score += self.end_transitions

        # Sum (log-sum-exp) over all possible tags
        # shape: (batch_size,)
        return torch.logsumexp(score, dim=1)

    def _viterbi_decode(self, emissions, mask):
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        assert mask[0].all()

        seq_length, batch_size = mask.shape

        # Start transition and first emission
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0]
        history = []

        # score is a tensor of size (batch_size, num_tags) where for every batch,
        # value at column j stores the score of the best tag sequence so far that ends
        # with tag j
        # history saves where the best tags candidate transitioned from; this is used
        # when we trace back the best tag sequence

        # Viterbi algorithm recursive case: we compute the score of the best tag sequence
        # for every possible next tag
        for i in range(1, seq_length):
            # Broadcast viterbi score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emission = emissions[i].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the score of the best
            # tag sequence so far that ends with transitioning from tag i to tag j and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emission

            # Find the maximum score over all possible current tag
            # shape: (batch_size, num_tags)
            next_score, indices = next_score.max(dim=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # and save the index that produces the next score
            # shape: (batch_size, num_tags)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)
            history.append(indices)

        # End transition score
        # shape: (batch_size, num_tags)
        score += self.end_transitions

        # Now, compute the best path for each sample

        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1
        best_tags_list = []

        for idx in range(batch_size):
            # Find the tag which maximizes the score at the last timestep; this is our best tag
            # for the last timestep
            _, best_last_tag = score[idx].max(dim=0)
            best_tags = [best_last_tag.item()]

            # We trace back where the best last tag comes from, append that to our best tag
            # sequence, and trace it back again, and so on
            for hist in reversed(history[:seq_ends[idx]]):
                best_last_tag = hist[idx][best_tags[-1]]
                best_tags.append(best_last_tag.item())

            # Reverse the order because we start from the last timestep
            best_tags.reverse()
            best_tags_list.append(best_tags)

        return best_tags_list


class NERNet(nn.Module):
    """
        NERNet : Lstm+CRF
    """

    def __init__(self, config):
        super(NERNet, self).__init__()
        char_emb = None#model_conf['char_emb']
        bichar_emb = None#model_conf['bichar_emb']
        embed_size = config.embed_size#args.char_emb_dim
        if char_emb is not None:
            # self.char_emb = nn.Embedding.from_pretrained(char_emb, freeze=False, padding_idx=0)

            self.char_emb = nn.Embedding(num_embeddings=char_emb.shape[0], embedding_dim=char_emb.shape[1],
                                         padding_idx=0, _weight=char_emb)
            self.char_emb.weight.requires_grad = True
            embed_size = char_emb.size()[1]
        else:
            vocab_size = config.vocab_size #len(model_conf['char_vocab'])
            self.char_emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size,
                                         padding_idx=0)
        self.bichar_emb = None
        if bichar_emb is not None:
            # self.bichar_emb = nn.Embedding.from_pretrained(bichar_emb, freeze=False, padding_idx=0)
            self.bichar_emb = nn.Embedding(num_embeddings=bichar_emb.shape[0], embedding_dim=bichar_emb.shape[1],
                                           padding_idx=0, _weight=bichar_emb)
            self.bichar_emb.weight.requires_grad = True

            embed_size += bichar_emb.size()[1]

        self.drop = nn.Dropout(p=config.dropout)
        # self.sentence_encoder = SentenceEncoder(args, embed_size)
        self.sentence_encoder = nn.LSTM(embed_size, config.hidden_size, num_layers=1, batch_first=True,
                                        bidirectional=True)
        self.emission = nn.Linear(config.hidden_size * 2, config.num_classes)
        self.crf = CRF(config.num_classes, batch_first=True)

    def forward(self, char_id, length, label_id=None):
        # use anti-mask for answers-locator
        # mask = char_id.eq(0)
        chars = self.char_emb(char_id)

        # if self.bichar_emb is not None:
        #     bichars = self.bichar_emb(bichar_id)
        #     chars = torch.cat([chars, bichars], dim=-1)
        chars = self.drop(chars)

        # sen_encoded = self.sentence_encoder(chars, mask)
        sen_encoded, _ = self.sentence_encoder(chars)
        sen_encoded = self.drop(sen_encoded)

        bio_mask = char_id != 0
        emission = self.emission(sen_encoded)
        emission = F.log_softmax(emission, dim=-1)

        if label_id is not None:
            crf_loss = -self.crf(emission, label_id, mask=bio_mask, reduction='mean')
            return crf_loss
        else:
            pred = self.crf.decode(emissions=emission, mask=bio_mask)
            # TODO:check
            max_len = char_id.size(1)
            temp_tag = copy.deepcopy(pred)
            for line in temp_tag:
                line.extend([0] * (max_len - len(line)))
            ent_pre = torch.tensor(temp_tag).to(emission.device)
            return ent_pre

class NERWNet(nn.Module):
    """
        NERNet : Lstm+CRF
    """

    def __init__(self, config):
        super(NERWNet, self).__init__()
        char_emb = None#model_conf['char_emb']
        bichar_emb = None#model_conf['bichar_emb']
        embed_size = config.embed_size#args.char_emb_dim
        if char_emb is not None:
            # self.char_emb = nn.Embedding.from_pretrained(char_emb, freeze=False, padding_idx=0)

            self.char_emb = nn.Embedding(num_embeddings=char_emb.shape[0], embedding_dim=char_emb.shape[1],
                                         padding_idx=0, _weight=char_emb)
            self.char_emb.weight.requires_grad = True
            embed_size = char_emb.size()[1]
        else:
            vocab_size = config.vocab_size #len(model_conf['char_vocab'])
            self.char_emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size,
                                         padding_idx=0)
        self.bichar_emb = None
        if bichar_emb is not None:
            # self.bichar_emb = nn.Embedding.from_pretrained(bichar_emb, freeze=False, padding_idx=0)
            self.bichar_emb = nn.Embedding(num_embeddings=bichar_emb.shape[0], embedding_dim=bichar_emb.shape[1],
                                           padding_idx=0, _weight=bichar_emb)
            self.bichar_emb.weight.requires_grad = True

            embed_size += bichar_emb.size()[1]

        self.kv = WordKVMN(config)

        self.drop = nn.Dropout(p=config.dropout)
        # self.sentence_encoder = SentenceEncoder(args, embed_size)
        self.sentence_encoder = nn.LSTM(embed_size, config.hidden_size, num_layers=1, batch_first=True,
                                        bidirectional=True)
        self.emission = nn.Linear(config.max_seq_len, config.num_classes)
        self.crf = CRF(config.num_classes, batch_first=True)

    def forward(self, char_id, length, label_id=None):
        # use anti-mask for answers-locator
        # mask = char_id.eq(0)
        chars = self.char_emb(char_id)

        # if self.bichar_emb is not None:
        #     bichars = self.bichar_emb(bichar_id)
        #     chars = torch.cat([chars, bichars], dim=-1)

        chars = self.drop(chars)

        # sen_encoded = self.sentence_encoder(chars, mask)
        sen_encoded, _ = self.sentence_encoder(chars)
        sen_encoded = self.drop(sen_encoded)

        sen_encoded = self.kv(sen_encoded, char_id, label_id)

        bio_mask = char_id != 0
        emission = self.emission(sen_encoded)
        emission = F.log_softmax(emission, dim=-1)

        if label_id is not None:
            crf_loss = -self.crf(emission, label_id, mask=bio_mask, reduction='mean')
            return crf_loss
        else:
            pred = self.crf.decode(emissions=emission, mask=bio_mask)
            # TODO:check
            max_len = char_id.size(1)
            temp_tag = copy.deepcopy(pred)
            for line in temp_tag:
                line.extend([0] * (max_len - len(line)))
            ent_pre = torch.tensor(temp_tag).to(emission.device)
            return ent_pre

class BERNet(nn.Module):
    """
        NERNet : Lstm+CRF
    """

    def __init__(self, config):
        super(BERNet, self).__init__()
        # char_emb = None#model_conf['char_emb']
        # bichar_emb = None#model_conf['bichar_emb']
        # embed_size = config.hidden_size#args.char_emb_dim
        # if char_emb is not None:
        #     # self.char_emb = nn.Embedding.from_pretrained(char_emb, freeze=False, padding_idx=0)
        #
        #     self.char_emb = nn.Embedding(num_embeddings=char_emb.shape[0], embedding_dim=char_emb.shape[1],
        #                                  padding_idx=0, _weight=char_emb)
        #     self.char_emb.weight.requires_grad = True
        #     embed_size = char_emb.size()[1]
        # else:
        #     vocab_size = config.vocab_size #len(model_conf['char_vocab'])
        #     self.char_emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=config.hidden_size,
        #                                  padding_idx=0)
        # self.bichar_emb = None
        # if bichar_emb is not None:
        #     # self.bichar_emb = nn.Embedding.from_pretrained(bichar_emb, freeze=False, padding_idx=0)
        #     self.bichar_emb = nn.Embedding(num_embeddings=bichar_emb.shape[0], embedding_dim=bichar_emb.shape[1],
        #                                    padding_idx=0, _weight=bichar_emb)
        #     self.bichar_emb.weight.requires_grad = True
        #
        #     embed_size += bichar_emb.size()[1]

        self.bert = BertModel.from_pretrained(config.bert_model_path)
        for param in self.bert.parameters():
            param.requires_grad = True

        self.drop = nn.Dropout(p=config.dropout)
        # self.sentence_encoder = SentenceEncoder(args, embed_size)
        self.sentence_encoder = nn.LSTM(config.hidden_size, config.sent_hidden_size, num_layers=1, batch_first=True,
                                        bidirectional=True)
        self.emission = nn.Linear(config.sent_hidden_size * 2, config.num_classes)
        self.crf = CRF(config.num_classes, batch_first=True)

    def forward(self, input_ids, attention_mask, token_type_ids, char_id, length, label_id=None):
        # use anti-mask for answers-locator
        # mask = char_id.eq(0)
        # chars = self.char_emb(char_id)
        _, _, layers = self.bert(input_ids, attention_mask, token_type_ids, output_hidden_states=True)
        chars = (layers[-1] + layers[0]) / 2
        #
        # # if self.bichar_emb is not None:
        # #     bichars = self.bichar_emb(bichar_id)
        # #     chars = torch.cat([chars, bichars], dim=-1)
        chars = self.drop(chars)

        # sen_encoded = self.sentence_encoder(chars, mask)
        # sen_encoded, _ = self.sentence_encoder(chars)
        sen_encoded = chars
        sen_encoded = self.drop(sen_encoded)

        bio_mask = char_id != 0
        emission = self.emission(sen_encoded)
        emission = F.log_softmax(emission, dim=-1)

        if label_id is not None:
            crf_loss = -self.crf(emission, label_id, mask=bio_mask, reduction='mean')
            return crf_loss
        else:
            pred = self.crf.decode(emissions=emission, mask=bio_mask)
            # TODO:check
            max_len = char_id.size(1)
            temp_tag = copy.deepcopy(pred)
            for line in temp_tag:
                line.extend([0] * (max_len - len(line)))
            ent_pre = torch.tensor(temp_tag).to(emission.device)
            return ent_pre

class RnnForSentencePairClassification(nn.Module):
    """Unidirectional GRU model for sentences pair classification.
    2 sentences use the same encoder and concat to a linear model.
    """
    def __init__(self, config):
        """Initialize the model with config dict.

        Args:
            config: python dict must contains the attributes below:
                config.vocab_size: vocab size
                config.hidden_size: RNN hidden size and embedding dim
                config.num_classes: int, e.g. 2
                config.dropout: float between 0 and 1
        """
        super().__init__()
        self.embedding = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=0)
        self.rnn = nn.GRU(
            config.hidden_size, hidden_size=config.hidden_size,
            bidirectional=False, batch_first=True)
        self.linear = nn.Linear(config.hidden_size , config.num_classes)
        self.dropout = nn.Dropout(config.dropout)
        self.num_classes = config.num_classes
        self.config = config

    def forward(self, s1_ids,  s1_lengths):
        """Forward inputs and get logits.

        Args:
            s1_ids: (batch_size, max_seq_len)
            s2_ids: (batch_size, max_seq_len)
            s1_lengths: (batch_size)
            s2_lengths: (batch_size)

        Returns:
            logits: (batch_size, num_classes)
        """
        batch_size = s1_ids.shape[0]
        # ids: (batch_size, max_seq_len)
        s1_embed = self.embedding(s1_ids)
        # s2_embed = self.embedding(s2_ids)
        # embed: (batch_size, max_seq_len, hidden_size)
        s1_packed: PackedSequence = pack_padded_sequence(
            s1_embed, s1_lengths, batch_first=True, enforce_sorted=False)
        # s2_packed: PackedSequence = pack_padded_sequence(
        #     s2_embed, s2_lengths, batch_first=True, enforce_sorted=False)
        # packed: (sum(lengths), hidden_size)
        self.rnn.flatten_parameters()
        s1_output, s1_hidden = self.rnn(s1_packed)
        s1_output, output_lengths = nn.utils.rnn.pad_packed_sequence(s1_output)
        seq_len = s1_output.shape[0]
        padding = (
            0, 0,
            0, 0,
            0,self.config.max_seq_len-seq_len
        )
        s1_output = torch.nn.functional.pad(s1_output,padding, value=-1)
        s1_output = s1_output.permute((1,0,2))

        # _, s2_hidden = self.rnn(s2_packed)
        # hidden = s1_hidden.view(batch_size, -1)
        hidden = self.linear(s1_output)
        # s2_hidden = s2_hidden.view(batch_size, -1)
        # hidden = torch.cat([s1_hidden, s2_hidden], dim=-1)
        # hidden = self.linear(hidden).view(-1, self.num_classes)
        hidden = self.dropout(hidden)
        logits = nn.functional.softmax(hidden, dim=-1)
        # logits: (batch_size, num_classes)
        return logits


class LogisticRegression(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=0)
        self.dropout = nn.Dropout(config.dropout)
        self.fc1 = nn.Linear(config.vocab_size, config.num_classes)

    def forward(self, s1_ids, s2_ids, s1_lengths, s2_lengths, **kwargs):
        batch_size = s1_ids.shape[0]
        s1_embed = self.embedding(s1_ids)
        s2_embed = self.embedding(s2_ids)
        # embed: (batch_size, max_seq_len, hidden_size)
        # s1_packed: PackedSequence = pack_padded_sequence(
        #     s1_embed, s1_lengths, batch_first=True, enforce_sorted=False)
        # s2_packed: PackedSequence = pack_padded_sequence(
        #     s2_embed, s2_lengths, batch_first=True, enforce_sorted=False)
        # _, s1_hidden = self.rnn(s1_packed)
        # _, s2_hidden = self.rnn(s2_packed)
        s1_hidden = s1_embed.view(batch_size, -1)
        s2_hidden = s2_embed.view(batch_size, -1)
        hidden = torch.cat([s1_hidden, s2_hidden], dim=-1)

        # x = torch.squeeze(hidden)  # (batch, vocab_size)
        x = self.dropout(hidden)
        logit = self.fc1(x)  # (batch, target_size)
        return logit



class CharCNN(nn.Module):

    def __init__(self, config):
        super().__init__()
        # self.is_cuda_enabled = config.cuda

        num_conv_filters = config.num_conv_filters
        output_channel = config.output_channel
        hidden_size = config.hidden_size
        target_class = config.num_classes
        input_channel = config.hidden_size

        self.embedding = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=0)

        self.conv1 = nn.Conv1d(input_channel, num_conv_filters, kernel_size=7)
        self.conv2 = nn.Conv1d(num_conv_filters, num_conv_filters, kernel_size=7)
        self.conv3 = nn.Conv1d(num_conv_filters, num_conv_filters, kernel_size=3)
        self.conv4 = nn.Conv1d(num_conv_filters, num_conv_filters, kernel_size=3)
        self.conv5 = nn.Conv1d(num_conv_filters, num_conv_filters, kernel_size=3)
        self.conv6 = nn.Conv1d(num_conv_filters, output_channel, kernel_size=3)

        self.dropout = nn.Dropout(config.dropout)
        self.fc1 = nn.Linear(output_channel, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, target_class)

    def forward(self, s1_ids, s2_ids, s1_lengths, s2_lengths):
        batch_size = s1_ids.shape[0]
        # ids: (batch_size, max_seq_len)
        s1_embed = self.embedding(s1_ids)
        s2_embed = self.embedding(s2_ids)

        embed = torch.cat([s1_embed, s2_embed], dim=1)
        # embed: (batch_size, max_seq_len, hidden_size)
        # s1_packed: PackedSequence = pack_padded_sequence(
        #     s1_embed, s1_lengths, batch_first=True, enforce_sorted=False)
        if torch.cuda.is_available():
            x = embed.transpose(1, 2).type(torch.cuda.FloatTensor)
            # x = embed.transpose(1, 2).type(torch.FloatTensor)
        else:
            x = embed.transpose(1, 2).type(torch.FloatTensor)

        x = F.max_pool1d(F.relu(self.conv1(x)), 3)
        x = F.max_pool1d(F.relu(self.conv2(x)), 3)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))

        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)
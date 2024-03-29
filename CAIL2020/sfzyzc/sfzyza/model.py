"""BERT and RNN model for sentence pair classification.

Author: Tsinghuaboy (tsinghua9boy@sina.com)

Used for SMP-CAIL2020-Argmine.
"""
import torch

from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence
from transformers import BertModel
from transformers import AutoModel
import torch.nn.functional as F
from capsnetx import PrimaryCaps, FCCaps, FlattenCaps

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
        self.linear = nn.Linear(config.hidden_size * 2, config.num_classes)
        self.dropout = nn.Dropout(config.dropout)
        self.num_classes = config.num_classes

    def forward(self, s1_ids, s2_ids, s1_lengths, s2_lengths):
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
        s2_embed = self.embedding(s2_ids)
        # embed: (batch_size, max_seq_len, hidden_size)
        s1_packed: PackedSequence = pack_padded_sequence(
            s1_embed, s1_lengths, batch_first=True, enforce_sorted=False)
        s2_packed: PackedSequence = pack_padded_sequence(
            s2_embed, s2_lengths, batch_first=True, enforce_sorted=False)
        # packed: (sum(lengths), hidden_size)
        self.rnn.flatten_parameters()
        _, s1_hidden = self.rnn(s1_packed)
        _, s2_hidden = self.rnn(s2_packed)
        s1_hidden = s1_hidden.view(batch_size, -1)
        s2_hidden = s2_hidden.view(batch_size, -1)
        hidden = torch.cat([s1_hidden, s2_hidden], dim=-1)
        hidden = self.linear(hidden).view(-1, self.num_classes)
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
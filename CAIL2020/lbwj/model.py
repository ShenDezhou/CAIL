"""BERT and RNN model for sentence pair classification.

Author: Tsinghuaboy (tsinghua9boy@sina.com)

Used for SMP-CAIL2020-Argmine.
"""
import torch

from torch import nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence
from transformers.modeling_bert import BertModel

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
        self.linear = nn.Linear(config.hidden_size, config.num_classes)
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
        # bert_output[1]: (batch_size, hidden_size)
        pooled_output = bert_output[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.linear(pooled_output).view(batch_size, self.num_classes)
        logits = self.bn(logits)
        logits = nn.functional.softmax(logits, dim=-1)
        # logits: (batch_size, num_classes)
        return logits

import torch.nn.functional as F
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

        num_conv_filters = config.num_conv_filters
        output_channel = config.output_channel
        hidden_size = config.num_fc_hidden_size
        target_class = config.num_classes
        input_channel = config.hidden_size
        # data(b, 512, 768) -> conv(b, 511,767) -> bn -> mp(b, 4, 6)
        self.conv1 = nn.Conv1d(input_channel, num_conv_filters, kernel_size=7)
        self.conv2 = nn.Conv1d(num_conv_filters, num_conv_filters, kernel_size=7)
        self.conv3 = nn.Conv1d(num_conv_filters, num_conv_filters, kernel_size=5)
        self.conv4 = nn.Conv1d(num_conv_filters, num_conv_filters, kernel_size=5)
        self.conv5 = nn.Conv1d(num_conv_filters, num_conv_filters, kernel_size=3)
        self.conv6 = nn.Conv1d(num_conv_filters, output_channel, kernel_size=3)

        #cnn feature map has a total number of 228 dimensions.
        self.dropout = nn.Dropout(config.dropout)
        self.fc1 = nn.Linear(output_channel, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, target_class)

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
        # encoded_output = bert_output[0]
        # # encoded_output[0]: (batch_size, 1, sequence_length, hidden_size)
        # encoded_output = encoded_output.view(batch_size, 1, encoded_output.shape[1], -1)

        # ids: (batch_size, max_seq_len)
        s1_embed = bert_output[0]
        # s2_embed = self.embedding(s2_ids)
        # embed: (batch_size, max_seq_len, hidden_size)
        # s1_packed: PackedSequence = pack_padded_sequence(
        #     s1_embed, s1_lengths, batch_first=True, enforce_sorted=False)
        if torch.cuda.is_available():
            x = s1_embed.transpose(1, 2).type(torch.cuda.FloatTensor)
        else:
            x = s1_embed.transpose(1, 2).type(torch.FloatTensor)

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
        logits = self.fc3(x)        # logits: (batch_size, num_classes)
        return logits

# from resnet import ResNet,BasicBlock,Bottleneck, resnet18,resnet34,resnet50,resnet101,resnet152, resnext50_32x4d, resnext101_32x8d, wide_resnet50_2, wide_resnet101_2
#
# resnet_pool = dict(zip(range(9),[resnet18,resnet34,resnet50,resnet101,resnet152, resnext50_32x4d, resnext101_32x8d, wide_resnet50_2, wide_resnet101_2]))
# from resnetv import ResNet
from resnetv import ResNet
from FPN import FPN
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

        hidden_size = config.fc_hidden
        target_class = config.num_classes
        # self.resnet = resnet18(num_classes=hidden_size)
        #self.resnet = ResNet(block=BasicBlock, layers=[1, 1, 1, 1], num_classes=hidden_size)
        self.resnet = ResNet(config.in_channels, 18)
        self.fpn = FPN([256]* 4, 4)

        self.fpn_seq = FPN([128,128,128,70], 4)
        #cnn feature map has a total number of 228 dimensions.
        self.dropout = nn.Dropout(config.dropout)
        self.fc1 = nn.Linear(hidden_size, target_class)
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
            output_hidden_states=True
        )
        # bert_output[0]: (batch_size, sequence_length, hidden_size)
        # encoded_output = bert_output[0]
        # # encoded_output[0]: (batch_size, 1, sequence_length, hidden_size)
        # encoded_output = encoded_output.view(batch_size, 1, encoded_output.shape[1], -1)

        # ids: (batch_size, max_seq_len)
        x = bert_output[2]
        # s2_embed = self.embedding(s2_ids)
        # embed: (batch_size, max_seq_len, hidden_size)
        # s1_packed: PackedSequence = pack_padded_sequence(
        #     s1_embed, s1_lengths, batch_first=True, enforce_sorted=False)
        # if torch.cuda.is_available():
        #     x = s1_embed.transpose(1, 2).type(torch.cuda.FloatTensor)
        # else:
        #     x = s1_embed.transpose(1, 2).type(torch.FloatTensor)
        #x = s1_embed.transpose(1, 2)
        x = [l.unsqueeze(1) for l in x[-3:]]
        x = torch.cat(x, dim=1)
        # x = self.resnet(x)
        x = x.permute((0,3,1,2))
        x = x[:,0:256,:,:], x[:,256:256+256,:,:], x[:,512:512+256,:,:], x[:,768:,:,:]
        x = self.fpn(x)

        x = x.permute((0, 3, 1, 2))
        x = x[:, 0:128, :, :], x[:, 128:128 + 128, :, :], x[:, 256:256 + 128, :, :], x[:, 384:, :, :]
        x = self.fpn_seq(x)
        x = x.flatten(start_dim=1)
        x = self.dropout(x)
        logits = self.fc1(x)        # logits: (batch_size, num_classes)
        return logits

from resnetv import ResNet
from FPN import FPN
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

        hidden_size = config.fc_hidden
        target_class = config.num_classes
        # self.resnet = resnet18(num_classes=hidden_size)
        #self.resnet = ResNet(block=BasicBlock, layers=[1, 1, 1, 1], num_classes=hidden_size)
        self.resnet = ResNet(config.in_channels, 18)
        self.fpn = FPN([256]* 4, 4)

        self.fpn_seq = FPN([128,128,128,70], 4)
        #cnn feature map has a total number of 228 dimensions.
        self.dropout = nn.Dropout(config.dropout)
        self.fc1 = nn.Linear(hidden_size, target_class)
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
            output_hidden_states=True
        )
        # bert_output[0]: (batch_size, sequence_length, hidden_size)
        # encoded_output = bert_output[0]
        # # encoded_output[0]: (batch_size, 1, sequence_length, hidden_size)
        # encoded_output = encoded_output.view(batch_size, 1, encoded_output.shape[1], -1)

        # ids: (batch_size, max_seq_len)
        x = bert_output[2]
        # s2_embed = self.embedding(s2_ids)
        # embed: (batch_size, max_seq_len, hidden_size)
        # s1_packed: PackedSequence = pack_padded_sequence(
        #     s1_embed, s1_lengths, batch_first=True, enforce_sorted=False)
        # if torch.cuda.is_available():
        #     x = s1_embed.transpose(1, 2).type(torch.cuda.FloatTensor)
        # else:
        #     x = s1_embed.transpose(1, 2).type(torch.FloatTensor)
        #x = s1_embed.transpose(1, 2)
        x = [l.unsqueeze(1) for l in x[-3:]]
        x = torch.cat(x, dim=1)
        # x = self.resnet(x)
        x = x.permute((0,3,1,2))
        x = x[:,0:256,:,:], x[:,256:256+256,:,:], x[:,512:512+256,:,:], x[:,768:,:,:]
        x = self.fpn(x)

        x = x.permute((0, 3, 1, 2))
        x = x[:, 0:128, :, :], x[:, 128:128 + 128, :, :], x[:, 256:256 + 128, :, :], x[:, 384:, :, :]
        x = self.fpn_seq(x)
        x = x.flatten(start_dim=1)
        x = self.dropout(x)
        logits = self.fc1(x)        # logits: (batch_size, num_classes)
        return logits

from resnetv import ResNet
from FPN import FPN
class BertYLForClassification(nn.Module):
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

        hidden_size = config.fc_hidden
        target_class = config.num_classes
        # self.resnet = resnet18(num_classes=hidden_size)
        #self.resnet = ResNet(block=BasicBlock, layers=[1, 1, 1, 1], num_classes=hidden_size)
        self.resnet = ResNet(config.in_channels, 18)
        self.fpn = FPN([256]* 4, 4)

        self.fpn_seq = FPN([128,128,128,70], 4)
        #cnn feature map has a total number of 228 dimensions.
        self.dropout = nn.Dropout(config.dropout)
        self.fc1 = nn.Linear(hidden_size, target_class)
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
            output_hidden_states=True
        )
        # bert_output[0]: (batch_size, sequence_length, hidden_size)
        # encoded_output = bert_output[0]
        # # encoded_output[0]: (batch_size, 1, sequence_length, hidden_size)
        # encoded_output = encoded_output.view(batch_size, 1, encoded_output.shape[1], -1)

        # ids: (batch_size, max_seq_len)
        x = bert_output[2]
        # s2_embed = self.embedding(s2_ids)
        # embed: (batch_size, max_seq_len, hidden_size)
        # s1_packed: PackedSequence = pack_padded_sequence(
        #     s1_embed, s1_lengths, batch_first=True, enforce_sorted=False)
        # if torch.cuda.is_available():
        #     x = s1_embed.transpose(1, 2).type(torch.cuda.FloatTensor)
        # else:
        #     x = s1_embed.transpose(1, 2).type(torch.FloatTensor)
        #x = s1_embed.transpose(1, 2)
        x = [l.unsqueeze(1) for l in x[-3:]]
        x = torch.cat(x, dim=1)
        # x = self.resnet(x)
        x = x.permute((0,3,1,2))
        x = x[:,0:256,:,:], x[:,256:256+256,:,:], x[:,512:512+256,:,:], x[:,768:,:,:]
        x = self.fpn(x)

        x = x.permute((0, 3, 1, 2))
        x = x[:, 0:128, :, :], x[:, 128:128 + 128, :, :], x[:, 256:256 + 128, :, :], x[:, 384:, :, :]
        x = self.fpn_seq(x)
        x = x.flatten(start_dim=1)
        x = self.dropout(x)
        logits = self.fc1(x)        # logits: (batch_size, num_classes)
        return logits

from FPN import FPN
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

        hidden_size = config.fc_hidden
        target_class = config.num_classes
        # self.resnet = resnet18(num_classes=hidden_size)
        #self.resnet = ResNet(block=BasicBlock, layers=[1, 1, 1, 1], num_classes=hidden_size)
        # self.resnet = ResNet(config.in_channels, 18)
        self.fpn = FPN([256]* 4, 4)

        self.fpn_seq = FPN([128,128,128,70], 4)
        #cnn feature map has a total number of 228 dimensions.
        self.dropout = nn.Dropout(config.dropout)
        self.fc1 = nn.Linear(hidden_size, target_class)
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
            output_hidden_states=True
        )
        # bert_output[0]: (batch_size, sequence_length, hidden_size)
        # encoded_output = bert_output[0]
        # # encoded_output[0]: (batch_size, 1, sequence_length, hidden_size)
        # encoded_output = encoded_output.view(batch_size, 1, encoded_output.shape[1], -1)

        # ids: (batch_size, max_seq_len)
        x = bert_output[2]
        # s2_embed = self.embedding(s2_ids)
        # embed: (batch_size, max_seq_len, hidden_size)
        # s1_packed: PackedSequence = pack_padded_sequence(
        #     s1_embed, s1_lengths, batch_first=True, enforce_sorted=False)
        # if torch.cuda.is_available():
        #     x = s1_embed.transpose(1, 2).type(torch.cuda.FloatTensor)
        # else:
        #     x = s1_embed.transpose(1, 2).type(torch.FloatTensor)
        #x = s1_embed.transpose(1, 2)
        x = [l.unsqueeze(1) for l in x[-3:]]
        x = torch.cat(x, dim=1)
        # x = self.resnet(x)
        x = x.permute((0,3,1,2))
        x = x[:,0:256,:,:], x[:,256:256+256,:,:], x[:,512:512+256,:,:], x[:,768:,:,:]
        x = self.fpn(x)

        x = x.permute((0, 3, 1, 2))
        x = x[:, 0:128, :, :], x[:, 128:128 + 128, :, :], x[:, 256:256 + 128, :, :], x[:, 384:, :, :]
        x = self.fpn_seq(x)
        x = x.flatten(start_dim=1)
        x = self.dropout(x)
        logits = self.fc1(x)        # logits: (batch_size, num_classes)
        return logits

class Attention(nn.Module):
    def __init__(self, conv_size, *args, **params):
        super(Attention, self).__init__()
        self.hidden_size = 1024
        self.fc = nn.Linear(conv_size, self.hidden_size)
        # self.rfc = nn.Linear(self.hidden_size, conv_size)

    def forward(self, x, y):
        bs = x.size()[0]
        x_ = self.fc(x) # x_ = x
        x_ = x_.view(bs, 4, -1)
        y = y.view(bs, 4, -1)

        y_ = torch.transpose(y, 1, 2)
        a_ = torch.bmm(x_, y_)

        x_atten = torch.softmax(a_, dim=2)
        x_atten = torch.bmm(x_atten, y)

        y_atten = torch.softmax(a_, dim=1)
        y_atten = torch.bmm(torch.transpose(y_atten, 2, 1), x_)

        x_atten = x_atten.view(bs, -1)
        # x_atten = self.rfc(x_atten)
        y_atten = y_atten.view(bs, -1)

        return x_atten, y_atten, a_

class BertZForClassification(nn.Module):
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

        self.output_dim = 10
        # data(b, 512, 768) -> conv(b, 511,767) -> bn -> mp(b, 4, 6)
        self.conv_module = nn.Sequential(
            nn.Conv2d(1, self.output_dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(self.output_dim),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(128, 128), stride=(128, 128), padding=(1, 1))
        )
        # data(b, 512, 768) -> conv(b, 255,255) -> bn -> mp(b, 4, 4)
        self.conv_module2 = nn.Sequential(
            nn.Conv2d(1, self.output_dim, kernel_size=(2,3), stride=(2,3),padding=(0,0)),
            nn.BatchNorm2d(self.output_dim),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(64, 64), stride=(64, 64),padding=(1,1))
        )
        # data(b, 512, 768) -> conv(b, 169, 192) -> bn -> mp(b, 5, 6)
        self.conv_module3 = nn.Sequential(
            nn.Conv2d(1, self.output_dim , kernel_size=(3, 4), stride=(3, 4), padding=(0, 0)),
            nn.BatchNorm2d(self.output_dim),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(32, 32), stride=(32, 32), padding=(1, 1))
        )
        # data(b, 512, 768) -> conv(b, 127, 127) -> bn -> mp(b, 4, 4)
        self.conv_module4 = nn.Sequential(
            nn.Conv2d(1, self.output_dim , kernel_size=(4, 6), stride=(4, 6), padding=(0, 0)),
            nn.BatchNorm2d(self.output_dim),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(32, 32), stride=(32, 32), padding=(1, 1))
        )
        # data(b, 512, 768) -> conv(b, 101, 108) -> bn -> mp(b, 6, 6)
        self.conv_module5 = nn.Sequential(
            nn.Conv2d(1, self.output_dim , kernel_size=(5, 7), stride=(5, 7), padding=(0, 0)),
            nn.BatchNorm2d(self.output_dim),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(16, 16), stride=(16, 16), padding=(1, 1))
        )
        # data(b, 512, 768) -> conv(b, 84, 84) -> bn -> mp(b, 5, 5)
        self.conv_module6 = nn.Sequential(
            nn.Conv2d(1, self.output_dim , kernel_size=(6, 9), stride=(6, 9), padding=(0, 0)),
            nn.BatchNorm2d(self.output_dim),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(16, 16), stride=(16, 16), padding=(1, 1))
        )
        # data(b, 512, 768) -> conv(b, 72, 75) -> bn -> mp(b, 9, 9)
        self.conv_module7 = nn.Sequential(
            nn.Conv2d(1, self.output_dim , kernel_size=(7, 10), stride=(7, 10), padding=(0, 0)),
            nn.BatchNorm2d(self.output_dim),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(8, 8), stride=(8, 8), padding=(1, 1))
        )
        self.conv_module8 = nn.Sequential(
            nn.Conv2d(1, self.output_dim , kernel_size=(8, 12), stride=(8, 12), padding=(0, 0)),
            nn.BatchNorm2d(self.output_dim),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(8, 8), stride=(8, 8), padding=(1, 1))
        )
        # data(b, 512, 768) -> conv(b, 255,255) -> bn -> mp(b, 4, 4)
        self.conv_module9 = nn.Sequential(
            nn.Conv2d(1, self.output_dim , kernel_size=(9, 13), stride=(9, 13), padding=(0, 0)),
            nn.BatchNorm2d(self.output_dim),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(8, 8), stride=(8, 8), padding=(1, 1))
        )
        # data(b, 512, 768) -> conv(b, 169, 192) -> bn -> mp(b, 5, 6)
        self.conv_moduleA = nn.Sequential(
            nn.Conv2d(1, self.output_dim , kernel_size=(10, 15), stride=(10, 15), padding=(0, 0)),
            nn.BatchNorm2d(self.output_dim),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4, 4), stride=(4, 4), padding=(1, 1))
        )
        # data(b, 512, 768) -> conv(b, 127, 127) -> bn -> mp(b, 4, 4)
        self.conv_moduleB = nn.Sequential(
            nn.Conv2d(1, self.output_dim , kernel_size=(11, 16), stride=(11, 16), padding=(0, 0)),
            nn.BatchNorm2d(self.output_dim),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4, 4), stride=(4, 4), padding=(1, 1))
        )
        # data(b, 512, 768) -> conv(b, 101, 108) -> bn -> mp(b, 6, 6)
        self.conv_moduleC = nn.Sequential(
            nn.Conv2d(1, self.output_dim , kernel_size=(12, 18), stride=(12, 18), padding=(0, 0)),
            nn.BatchNorm2d(self.output_dim),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(1, 1))
        )
        # data(b, 512, 768) -> conv(b, 84, 84) -> bn -> mp(b, 5, 5)
        self.conv_moduleD = nn.Sequential(
            nn.Conv2d(1, self.output_dim , kernel_size=(13, 19), stride=(13, 19), padding=(0, 0)),
            nn.BatchNorm2d(self.output_dim),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(1, 1))
        )
        # data(b, 512, 768) -> conv(b, 72, 75) -> bn -> mp(b, 9, 9)
        self.conv_moduleE = nn.Sequential(
            nn.Conv2d(1, self.output_dim , kernel_size=(14, 21), stride=(14, 21), padding=(0, 0)),
            nn.BatchNorm2d(self.output_dim),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(1, 1))
        )
        self.att = Attention(14870)
        #cnn feature map has a total number of 228 dimensions.
        self.dropout = nn.Dropout(config.dropout)
        # 1-7: 228; 8-14: 1691
        self.linear = nn.Linear(config.hidden_size * 2, config.num_classes)
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
        cnn, bert, att = self.att(con_cnn_feats, pooled_output)

        pooled_output = torch.cat([cnn, bert], dim=1)
        pooled_output = self.dropout(pooled_output)
        logits = self.linear(pooled_output).view(batch_size, self.num_classes)
        #logits = self.bn(logits)
        logits = nn.functional.softmax(logits, dim=-1)
        # logits: (batch_size, num_classes)
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

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.encoder.LSTMEncoder import LSTMEncoder
from model.layer.Attention import Attention
from tools.accuracy_tool import single_label_top1_accuracy
from model.qa.util import generate_ans


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

from model.encoder.GRUEncoder import GRUEncoder
from model.qa.resnet import ResNet,BasicBlock
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

        # input_channel = self.hidden_size
        # num_conv_filters = config.getint("model", "num_conv_filters")#config.num_conv_filters
        # output_channel = config.getint("model", "output_channel")#config.output_channel

        #
        # self.conv1 = nn.Conv1d(input_channel, num_conv_filters, kernel_size=7)
        # self.conv2 = nn.Conv1d(num_conv_filters, num_conv_filters, kernel_size=7)
        # self.conv3 = nn.Conv1d(num_conv_filters, num_conv_filters, kernel_size=3)
        # self.conv4 = nn.Conv1d(num_conv_filters, num_conv_filters, kernel_size=3)
        # self.conv5 = nn.Conv1d(num_conv_filters, num_conv_filters, kernel_size=3)
        # self.conv6 = nn.Conv1d(num_conv_filters, output_channel, kernel_size=3)
        # self.dropout = nn.Dropout(config.getfloat("model", "dropout"))
        # self.fc1 = nn.Linear(output_channel, hidden_size)


        self.rank_module = nn.Linear(hidden_size, 1)

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

        # x = torch.cat([torch.max(c, dim=1)[0], torch.max(q, dim=1)[0]], dim=1)
        x = torch.cat([c, q], dim=1)
        if torch.cuda.is_available():
            x = x.transpose(1, 2).type(torch.cuda.FloatTensor)
        else:
            x = x.transpose(1, 2).type(torch.FloatTensor)

        x = self.resnet(x)
        # x = F.max_pool1d(F.relu(self.conv1(x)), 3)
        # x = F.max_pool1d(F.relu(self.conv2(x)), 3)
        # x = F.relu(self.conv3(x))
        # x = F.relu(self.conv4(x))
        # x = F.relu(self.conv5(x))
        # x = F.relu(self.conv6(x))
        #
        # x = F.max_pool1d(x, x.size(2)).squeeze(2).view(x.size(0), -1)
        # # x = F.relu(self.fc1(x.view(x.size(0), -1)))
        # x = self.dropout(x)

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


from model.encoder.GRUEncoder import GRUEncoder
class GRUModel(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(GRUModel, self).__init__()

        self.hidden_size = config.getint("model", "hidden_size")
        self.word_num = 0
        f = open(config.get("data", "word2id"), "r", encoding="utf8")
        for line in f:
            self.word_num += 1

        self.embedding = nn.Embedding(self.word_num, self.hidden_size)
        self.context_encoder = GRUEncoder(config, gpu_list, *args, **params)
        self.question_encoder = GRUEncoder(config, gpu_list, *args, **params)
        self.attention = Attention(config, gpu_list, *args, **params)

        input_channel = self.hidden_size
        num_conv_filters = config.getint("model", "num_conv_filters")#config.num_conv_filters
        output_channel = config.getint("model", "output_channel")#config.output_channel
        # hidden_size = config.getint("model", "num_fc_hidden_size")#config.num_fc_hidden_size

        self.conv1 = nn.Conv1d(input_channel, num_conv_filters, kernel_size=7)
        self.conv2 = nn.Conv1d(num_conv_filters, num_conv_filters, kernel_size=7)
        self.conv3 = nn.Conv1d(num_conv_filters, num_conv_filters, kernel_size=3)
        self.conv4 = nn.Conv1d(num_conv_filters, num_conv_filters, kernel_size=3)
        self.conv5 = nn.Conv1d(num_conv_filters, num_conv_filters, kernel_size=3)
        self.conv6 = nn.Conv1d(num_conv_filters, output_channel, kernel_size=3)
        self.dropout = nn.Dropout(config.getfloat("model", "dropout"))
        # self.fc1 = nn.Linear(output_channel, hidden_size)


        self.rank_module = nn.Linear(output_channel, 1)

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

        # x = torch.cat([torch.max(c, dim=1)[0], torch.max(q, dim=1)[0]], dim=1)
        x = torch.cat([c, q], dim=1)
        if torch.cuda.is_available():
            x = x.transpose(1, 2).type(torch.cuda.FloatTensor)
        else:
            x = x.transpose(1, 2).type(torch.FloatTensor)
        x = F.max_pool1d(F.relu(self.conv1(x)), 3)
        x = F.max_pool1d(F.relu(self.conv2(x)), 3)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))

        x = F.max_pool1d(x, x.size(2)).squeeze(2).view(x.size(0), -1)
        # x = F.relu(self.fc1(x.view(x.size(0), -1)))
        x = self.dropout(x)

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

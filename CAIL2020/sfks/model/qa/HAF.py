import torch
import torch.nn as nn
import torch.nn.functional as F

from model.encoder.GRUEncoder import GRUEncoder
# from model.layer.Attention import Attention
from tools.accuracy_tool import single_label_top1_accuracy
from model.qa.util import generate_ans

"""class BiAttention(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(BiAttention, self).__init__()

        self.attention = Attention(config, gpu_list, *args, **params)

    def init_multi_gpu(self, device, config, *args, **params):
        pass

    def forward(self, x1, x2):
        c, q, a = self.attention(x1, x2)

        y = torch.cat([torch.max(c, dim=1)[0], torch.max(q, dim=1)[0]], dim=1)

        return y
        """


class Attention(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(Attention, self).__init__()

        self.hidden_size = config.getint("model", "hidden_size")
        self.fc = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, x, y):
        x_ = self.fc(x)
        y_ = torch.transpose(y, 1, 2)
        a_ = torch.bmm(x_, y_)

        s = torch.softmax(a_, dim=2)
        a = torch.mean(s, dim=1)

        return a


class HAF(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(HAF, self).__init__()

        self.hidden_size = config.getint("model", "hidden_size")
        self.word_num = 0
        f = open(config.get("data", "word2id"), "r", encoding="utf8")
        for line in f:
            self.word_num += 1

        self.embedding = nn.Embedding(self.word_num, self.hidden_size)

        self.question_encoder = GRUEncoder(config, gpu_list, *args, **params)
        self.passage_encoder = GRUEncoder(config, gpu_list, *args, **params)
        self.option_encoder = GRUEncoder(config, gpu_list, *args, **params)
        self.s = GRUEncoder(config, gpu_list, *args, **params)

        self.q2p = Attention(config, gpu_list, *args, **params)
        self.q2o = Attention(config, gpu_list, *args, **params)
        self.o2p = Attention(config, gpu_list, *args, **params)
        self.oc = Attention(config, gpu_list, *args, **params)

        self.wp = nn.Linear(self.hidden_size, self.hidden_size * 2)
        self.score = nn.Linear(
            config.getint("data", "topk") * config.getint("data", "option_len") * config.getint("data", "passage_len"),
            1)

        self.criterion = nn.CrossEntropyLoss()

        self.multi = config.getboolean("data", "multi_choice")
        self.multi_module = nn.Linear(4, 16)
        self.accuracy_function = single_label_top1_accuracy

    def init_multi_gpu(self, device, config, *args, **params):
        pass
        # self.bert = nn.DataParallel(self.bert, device_ids=device)

    def forward(self, data, config, gpu_list, acc_result, mode):
        passage = data["passage"]
        question = data["question"]
        option = data["option"]

        batch = question.size()[0]
        option_num = option.size()[1]
        k = config.getint("data", "topk")

        passage = passage.view(batch * option_num * k, -1)
        question = question.view(batch, -1)
        option = option.view(batch * option_num, -1)
        # print(passage.size(), question.size(), option.size())

        passage = self.embedding(passage)
        question = self.embedding(question)
        option = self.embedding(option)
        # print(passage.size(), question.size(), option.size())

        _, passage = self.passage_encoder(passage)
        _, question = self.question_encoder(question)
        _, option = self.option_encoder(option)
        # print(passage.size(), question.size(), option.size())

        passage = passage.view(batch * option_num * k, -1, self.hidden_size)
        question = question.view(batch, 1, 1, -1, self.hidden_size).repeat(1, option_num, k, 1, 1).view(
            batch * option_num * k, -1, self.hidden_size)
        option = option.view(batch, option_num, 1, -1, self.hidden_size).repeat(1, 1, k, 1, 1).view(
            batch * option_num * k, -1, self.hidden_size)
        # print(passage.size(), question.size(), option.size())

        vp = self.q2p(question, passage).view(batch * option_num * k, -1, 1)
        # print("vp", vp.size())
        vp = vp * passage
        # print("vp", vp.size())
        vo = self.q2o(question, option).view(batch * option_num * k, -1, 1)
        # print("vo", vo.size())
        vo = vo * option
        # print("vo", vo.size())
        _, vpp = self.s(vp)
        # print("vpp", vpp.size())
        rp = self.q2o(vo, vpp).view(batch * option_num * k, -1, 1)
        # print("rp", rp.size())
        rp = rp * vpp
        # print("rp", rp.size())
        vop = self.oc(vo, vo).view(batch * option_num * k, -1, 1)
        # print("vop", vop.size())
        vop = vop * vo
        # print("vop", vop.size())
        ro = torch.cat([vo, vo - vop], dim=2)
        # print("ro", ro.size())

        s = self.wp(rp)
        ro = torch.transpose(ro, 2, 1)
        s = torch.bmm(s, ro)
        # print(s.size())
        s = s.view(batch * option_num, -1)
        s = self.score(s)
        y = s.view(batch, option_num)
        # print(y.size())
        # gg

        if self.multi:
            y = self.multi_module(y)

        if mode != "test":
            label = data["label"]
            loss = self.criterion(y, label)
            acc_result = self.accuracy_function(y, label, config, acc_result)
            return {"loss": loss, "acc_result": acc_result}

        return {"output": generate_ans(data["id"], y)}

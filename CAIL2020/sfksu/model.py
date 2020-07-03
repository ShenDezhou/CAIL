import torch
import torch.nn as nn
import torch.nn.functional as F
# from pytorch_pretrained_bert import BertModel
from transformers.modeling_bert import BertModel
from accuracy_tool import single_label_top1_accuracy


class BertQA(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(BertQA, self).__init__()

        self.bert = BertModel.from_pretrained(config.get("model", "bert_path"))
        # print(self.bert)
        # self.rank_module = nn.Linear(768 * config.getint("data", "topk"), 1)
        self.criterion = nn.CrossEntropyLoss()
        self.multi = config.getboolean("data", "multi_choice")
        self.multi_module = nn.Linear(768, 15)
        self.softmax = nn.Softmax(dim=-1)
        self.accuracy_function = single_label_top1_accuracy

    def init_multi_gpu(self, device, config, *args, **params):
        self.bert = nn.DataParallel(self.bert, device_ids=device)

    def forward(self, data, config, gpu_list, acc_result, mode):
        text = data["text"]
        token = data["token"]
        mask = data["mask"]

        batch = text.size()[0]
        option = text.size()[1]
        k = config.getint("data", "topk")
        option = option // k

        text = text.view(text.size()[0] * text.size()[1], text.size()[2])
        token = token.view(token.size()[0] * token.size()[1], token.size()[2])
        mask = mask.view(mask.size()[0] * mask.size()[1], mask.size()[2])

        _, y = self.bert.forward(text, token, mask)
        y = y.view(batch, -1)
        # y = self.rank_module(y)
        # y = y.view(batch, option)
        y = self.multi_module(y)
        y = self.softmax(y)
        label = data["label"]

        loss = self.criterion(y, label)
        acc_result = self.accuracy_function(y, label, config, acc_result)

        answer = []
        predm = y.argmax(dim=1)
        for x in range(batch):
            i = predm[x]
            if i == 0:
                subanswer = ['A']
            if i == 1:
                subanswer = ['B']
            if i == 2:
                subanswer = ['C']
            if i == 3:
                subanswer = ['D']
            if i == 4:
                subanswer = ['A', 'B']
            if i == 5:
                subanswer = ['A', 'C']
            if i == 6:
                subanswer = ['B', 'C']
            if i == 7:
                subanswer = ['A', 'B', 'C']
            if i == 8:
                subanswer = ['A', 'D']
            if i == 9:
                subanswer = ['B', 'D']
            if i == 10:
                subanswer = ['A', 'B', 'D']
            if i == 11:
                subanswer = ['C', 'D']
            if i == 12:
                subanswer = ['A', 'C', 'D']
            if i == 13:
                subanswer = ['B', 'C', 'D']
            if i == 14:
                subanswer = ['A', 'B', 'C', 'D']
            answer.append(subanswer)
        output = [{"id": id, "answer": [answer]} for id, answer in zip(data['id'], answer)]

        return {"loss": loss, "acc_result": acc_result, "output": output}

#20200621, bert -> dropout -> conv -> linear -> bn -> softmax: acc=14.42

class BertXQA(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(BertXQA, self).__init__()
        self.bert = BertModel.from_pretrained(config.get("model", "bert_path"))
        self.dropout = nn.Dropout(0.2)
        self.criterion = nn.CrossEntropyLoss()
        # self.multi = config.getboolean("data", "multi_choice")
        # self.multi_module = nn.Linear(4, 15)
        # self.softmax = nn.Softmax(dim=-1)
        # (b, 4, 768) -> conv(b, 4, 768) -> mp(b, 3, 6)
        self.conv_module = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 32), stride=(2, 32), padding=(1, 1))
        )
        self.linear = nn.Linear(18, config.getint("model", "num_classes"))

        self.accuracy_function = single_label_top1_accuracy
        self.bn = nn.BatchNorm1d(config.getint("model", "num_classes"))
        self.num_classes = config.getint("model", "num_classes")

    def init_multi_gpu(self, device, config, *args, **params):
        self.bert = nn.DataParallel(self.bert, device_ids=device)

    def forward(self, data, config, gpu_list, acc_result, mode):
        text = data["text"]
        token = data["token"]
        mask = data["mask"]

        batch_size = text.size()[0]
        option = text.size()[1]
        k = config.getint("data", "topk")
        option = option // k

        text = text.view(text.size()[0] * text.size()[1], text.size()[2])
        token = token.view(token.size()[0] * token.size()[1], token.size()[2])
        mask = mask.view(mask.size()[0] * mask.size()[1], mask.size()[2])

        bert_output = self.bert.forward(text, token, mask
                                      # , output_all_encoded_layers=False
                                     )
        # bert_output[0]: (batch_size, sequence_length, hidden_size)
        # bert_output[1]: (batch_size, hidden_size)
        pooled_output = bert_output[1]
        pooled_output = pooled_output.view(batch_size, 1, option, -1)
        pooled_output = self.dropout(pooled_output)
        conv_output = self.conv_module(pooled_output)
        conv_output = conv_output.reshape((batch_size, -1))
        logits = self.linear(conv_output)
        logits = self.bn(logits)
        logits = nn.functional.softmax(logits, dim=-1)
        label = data["label"]
        loss = self.criterion(logits, label)
        acc_result = self.accuracy_function(logits, label, config, acc_result)

        answer = []
        predm = logits.argmax(dim=1)
        preds = logits[:, 0:3].argmax(dim=1)
        for x in range(batch_size):
            if data['sorm'][x]:
                i = predm[x]
                subanswer = []
                if i == 0:
                    subanswer.append('A')
                if i == 1:
                    subanswer.append('B')
                if i == 2:
                    subanswer.append('C')
                if i == 3:
                    subanswer.append('D')
                if i == 4:
                    subanswer = ['A', 'B']
                if i == 5:
                    subanswer = ['A', 'C']
                if i == 6:
                    subanswer = ['B', 'C']
                if i == 7:
                    subanswer = ['A', 'B', 'C']
                if i == 8:
                    subanswer = ['A', 'D']
                if i == 9:
                    subanswer = ['B', 'D']
                if i == 10:
                    subanswer = ['A', 'B', 'D']
                if i == 11:
                    subanswer = ['C', 'D']
                if i == 12:
                    subanswer = ['A', 'C', 'D']
                if i == 13:
                    subanswer = ['B', 'C', 'D']
                if i == 14:
                    subanswer = ['A', 'B', 'C', 'D']
                answer.append(subanswer)
            else:
                answer.append(chr(ord('A') + preds[x]))
        output = [{"id": id, "answer": [answer]} for id, answer in zip(data['id'], answer)]
        return {"loss": loss, "acc_result": acc_result, "output": output}

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert import BertModel

from tools.accuracy_tool import single_label_top1_accuracy

def conv3x3(in_planes, out_planes, stride=1):#基本的3x3卷积
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BertQACNN(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(BertQACNN, self).__init__()

        self.bert = BertModel.from_pretrained(config.get("model", "bert_path"))
        # print(self.bert)
        # self.rank_module = nn.Linear(768 * config.getint("data", "topk"), 1)
        self.rank_module = nn.Linear(36, 60)
        self.rank_module2 = nn.Linear(60, 4)

        self.criterion = nn.CrossEntropyLoss()

        self.multi = config.getboolean("data", "multi_choice")
        self.multi_module = nn.Linear(60, 15)
        self.accuracy_function = single_label_top1_accuracy

        p = 3
        q = 16  # 8 * 8 * 4 * 3
        r = 120
        # q = 768 * config.getint("data", "topk")  # 8 * 4 * 3
        # q = 768 * config.getint("data", "topk")  # 4 * 3
        self.conv1 = nn.Conv2d(4, p, kernel_size=256, stride=1, padding=1, bias=False)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # C1中唯一出现了一次Maxpooling
        self.bn1 = nn.BatchNorm2d(p)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(p, q, kernel_size=3, stride=1, padding=1, bias=False)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # C1中唯一出现了一次Maxpooling
        self.bn2 = nn.BatchNorm2d(q)
        self.conv2 = nn.Conv2d(q, r, kernel_size=3, stride=1, padding=1, bias=False)
        self.single_module = nn.Linear(r, 4)
        # self.downsample = True
        # self.stride = 1
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # C1中唯一出现了一次Maxpooling


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

        # print('first:',text, token, mask, batch, option, k)
        text = text.view(text.size()[0] * text.size()[1], text.size()[2])
        token = token.view(token.size()[0] * token.size()[1], token.size()[2])
        mask = mask.view(mask.size()[0] * mask.size()[1], mask.size()[2])
        print(batch)
        # print('second', text, token, mask, batch, option, k)
        encode, y = self.bert.forward(text, token, mask, output_all_encoded_layers=False)
        # print('last', text,token,mask, batch, option, k)
        l = encode.size()[1]
        p = encode.size()[2]
        # y = y.view(batch  * option, l, -1)
        # #y = self.rank_module(y)
        encode = encode.view(3, option, l, l)

        y = self.conv1(encode)
        y = self.maxpool1(y)
        y = self.bn1(y)
        y = self.relu(y)

        y = y.view(batch, option * 9)
        y = self.rank_module(y)


        if data['sorm'][0]:
            y = self.multi_module(y)
        else:
            y = y.view(batch, option)
            y = self.rank_module2(y)

        label = data["label"]
        loss = self.criterion(y, label)
        acc_result = self.accuracy_function(y, label, config, acc_result)


        # if config.getboolean("data", "multi_choice"):
        if data['sorm'][0]:
            ind = y.argmax(dim=1) + 1
            answer = []
            for i in ind:
                subanswer = []
                # if i==1:
                #     subanswer.append('A')
                # if i==2:
                #     subanswer.append('B')
                # if i==4:
                #     subanswer.append('C')
                # if i==8:
                #     subanswer.append('D')
                if i == 1:
                    subanswer=['A', 'B']
                if i == 2:
                    subanswer = ['A','C']
                if i == 3:
                    subanswer = ['B','C']
                if i == 4:
                    subanswer = ['A','B','C']
                if i == 5:
                    subanswer = ['A','D']
                if i == 6:
                    subanswer = ['B','D']
                if i == 7:
                    subanswer = ['A','B','D']
                if i == 8:
                    subanswer = ['C','D']
                if i == 9:
                    subanswer = ['A','C','D']
                if i == 10:
                    subanswer = ['B','C','D']
                if i == 11:
                    subanswer = ['A','B','C','D']
                if i==12:
                    subanswer.append('A')
                if i==13:
                    subanswer.append('B')
                if i==14:
                    subanswer.append('C')
                if i==15:
                    subanswer.append('D')

                answer.append(subanswer)
            output = [{"id": id, "answer": [answer]} for id, answer in zip(data['id'], answer)]
        else:
            ind = y.argmax(dim=1)
            answer=[]
            for i in ind:
                answer.append(chr(ord('A')+i))
            output = [{"id": id, "answer": [answer]} for id, answer in zip(data['id'], answer)]

        return {"loss": loss, "acc_result": acc_result, "output": output}

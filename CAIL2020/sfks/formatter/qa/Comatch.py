import json
import torch
import numpy as np
import jieba
import random


class ComatchingFormatter:
    def __init__(self, config, mode):
        self.word2id = json.load(open(config.get("data", "word2id"), "r"))

        self.sent_max_len = config.getint("data", "sent_max_len")
        self.max_sent = config.getint("data", "max_sent")
        self.k = config.getint("data", "topk")

        self.symbol = [",", ".", "?", "\"", "”", "。", "？", ""]
        self.last_symbol = [".", "?", "。", "？"]

    def transform(self, word):
        if not (word in self.word2id.keys()):
            return self.word2id["UNK"]
        else:
            return self.word2id[word]

    def seq2tensor(self, sents, max_len):
        sent_len_max = max([len(s) for s in sents])
        sent_len_max = min(sent_len_max, max_len)

        sent_tensor = torch.LongTensor(len(sents), sent_len_max).zero_()

        sent_len = torch.LongTensor(len(sents)).zero_()
        for s_id, sent in enumerate(sents):
            sent_len[s_id] = len(sent)
            for w_id, word in enumerate(sent):
                if w_id >= sent_len_max: break
                sent_tensor[s_id][w_id] = self.transform(word)
        return [sent_tensor, sent_len]

    def seq2Htensor(self, docs, max_sent, max_sent_len, v1=0, v2=0):
        sent_num_max = max([len(s) for s in docs])
        sent_num_max = min(sent_num_max, max_sent)
        sent_len_max = max([len(w) for s in docs for w in s])
        sent_len_max = min(sent_len_max, max_sent_len)
        sent_num_max = max(sent_num_max, v1)
        sent_len_max = max(sent_len_max, v2)

        sent_tensor = torch.LongTensor(len(docs), sent_num_max, sent_len_max).zero_()
        sent_len = torch.LongTensor(len(docs), sent_num_max).zero_()
        doc_len = torch.LongTensor(len(docs)).zero_()
        for d_id, doc in enumerate(docs):
            doc_len[d_id] = len(doc)
            for s_id, sent in enumerate(doc):
                if s_id >= sent_num_max: break
                sent_len[d_id][s_id] = len(sent)
                for w_id, word in enumerate(sent):
                    if w_id >= sent_len_max: break
                    sent_tensor[d_id][s_id][w_id] = self.transform(word)
        return [sent_tensor, doc_len, sent_len]

    def gen_max(self, docs, max_sent, max_sent_len):
        sent_num_max = max([len(s) for s in docs])
        sent_num_max = min(sent_num_max, max_sent)
        sent_len_max = max([len(w) for s in docs for w in s])
        sent_len_max = min(sent_len_max, max_sent_len)

        return sent_num_max, sent_len_max

    def parse(self, sent):
        result = []
        for word in sent:
            if len(word) == 0:
                continue

            result.append(word)

        return result

    def parseH(self, sent):
        result = []
        temp = []
        for word in sent:
            temp.append(word)
            last = False
            for symbol in self.last_symbol:
                if word == symbol:
                    last = True
            if last:
                result.append(temp)
                temp = []

        if len(temp) != 0:
            result.append(temp)

        return result

    def process(self, data, config, mode, *args, **params):
        document = [[], [], [], []]
        option = []
        question = []
        label = []

        for temp_data in data:
            question.append(self.parse(temp_data["statement"]))

            if config.getboolean("data", "multi_choice"):
                option.append([self.parse(temp_data["option_list"]["A"]),
                               self.parse(temp_data["option_list"]["B"]),
                               self.parse(temp_data["option_list"]["C"]),
                               self.parse(temp_data["option_list"]["D"])])

                label_x = 0
                if "A" in temp_data["answer"]:
                    label_x += 1
                if "B" in temp_data["answer"]:
                    label_x += 2
                if "C" in temp_data["answer"]:
                    label_x += 4
                if "D" in temp_data["answer"]:
                    label_x += 8
            else:
                option.append([self.parse(temp_data["option_list"]["A"]),
                               self.parse(temp_data["option_list"]["B"]),
                               self.parse(temp_data["option_list"]["C"]),
                               self.parse(temp_data["option_list"]["D"])])

                label_x = 0
                if "A" in temp_data["answer"]:
                    label_x = 0
                if "B" in temp_data["answer"]:
                    label_x = 1
                if "C" in temp_data["answer"]:
                    label_x = 2
                if "D" in temp_data["answer"]:
                    label_x = 3

            temp = []
            for a in range(0, 4):
                arr = ["A", "B", "C", "D"]
                res = []
                k = [0, 1, 2, 6, 12, 7, 13, 3, 8, 9, 14, 15, 4, 10, 16, 5, 16, 17]
                for b in range(0, self.k):
                    res.append(self.parseH(temp_data["reference"][arr[a]][k[b]]))
                document[a].append(res)

            label.append(label_x)

        v1 = 0
        v2 = 0
        for a in range(0, 4):
            for b in range(0, len(document[a])):
                v1t, v2t = self.gen_max(document[a][b], self.max_sent, self.sent_max_len)
                v1 = max(v1, v1t)
                v2 = max(v2, v2t)
        option = self.seq2Htensor(option, self.max_sent, self.sent_max_len)
        question = self.seq2tensor(question, self.sent_max_len)

        for a in range(0, 4):
            for b in range(0, len(document[a])):
                document[a][b] = self.seq2Htensor(document[a][b], self.max_sent, self.sent_max_len, v1, v2)

        document_sent = []
        document_len = []
        do = []
        for a in range(0, 4):
            d = []
            ds = []
            dl = []
            for b in range(0, len(document[a])):
                d.append(document[a][b][0])
                ds.append(document[a][b][1])
                dl.append(document[a][b][2])

            d = torch.stack(d)
            ds = torch.stack(ds)
            dl = torch.stack(dl)

            do.append(d)
            document_sent.append(ds)
            document_len.append(dl)

        document = torch.stack(do)
        document_len = torch.stack(document_len)
        document_sent = torch.stack(document_sent)

        document = torch.transpose(document, 1, 0)
        document_len = torch.transpose(document_len, 1, 0)
        document_sent = torch.transpose(document_sent, 1, 0)

        label = torch.tensor(label, dtype=torch.long)

        return {
            "question": question[0],
            "question_len": question[1],
            "option": option[0],
            "option_sent": option[1],
            "option_len": option[2],
            "document": document,
            "document_sent": document_sent,
            "document_len": document_len,
            "label": label
        }

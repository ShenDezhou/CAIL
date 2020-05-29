import json
import torch
import numpy as np
import os
from pytorch_pretrained_bert import BertTokenizer


class CharQA:
    def __init__(self, config, mode):
        self.max_len1 = config.getint("data", "max_len1")
        self.max_len2 = config.getint("data", "max_len2")

        self.tokenizer = BertTokenizer.from_pretrained(config.get("data", "word2id"))
        self.k = config.getint("data", "topk")

    def convert(self, tokens, l):
        tokens = self.tokenizer.tokenize(tokens)
        while len(tokens) < l:
            tokens.append("[PAD]")
        tokens = tokens[:l]
        ids = self.tokenizer.convert_tokens_to_ids(tokens)

        return ids

    def process(self, data, config, mode, *args, **params):
        context = []
        question = []
        label = []

        for temp_data in data:
            if config.getboolean("data", "multi_choice"):
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
                label_x = 0
                if "A" in temp_data["answer"]:
                    label_x = 0
                if "B" in temp_data["answer"]:
                    label_x = 1
                if "C" in temp_data["answer"]:
                    label_x = 2
                if "D" in temp_data["answer"]:
                    label_x = 3

            label.append(label_x)

            temp_context = []
            temp_question = []

            for option in ["A", "B", "C", "D"]:
                res = temp_data["statement"] + temp_data["option_list"][option]
                text = []
                temp_question.append(self.convert(res, self.max_len1))

                ref = []
                k = [0, 1, 2, 6, 12, 7, 13, 3, 8, 9, 14, 15, 4, 10, 16, 5, 16, 17]
                for a in range(0, self.k):
                    res = temp_data["reference"][option][k[a]]

                    ref.append(self.convert(res, self.max_len2))

                temp_context.append(ref)

            context.append(temp_context)
            question.append(temp_question)

        question = torch.LongTensor(question)
        context = torch.LongTensor(context)
        label = torch.LongTensor(np.array(label, dtype=np.int32))

        return {"context": context, "question": question, 'label': label}

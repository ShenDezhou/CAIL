import json
import torch
import numpy as np
import os
# from pytorch_pretrained_bert import BertTokenizer
from tqdm import tqdm
from transformers import BertTokenizer
# from gbt.SingleMulti import SingleMulti

class Data:
    def __init__(self, config, mode):
        self.max_len1 = config.getint("data", "max_len1")
        self.max_len2 = config.getint("data", "max_len2")

        self.tokenizer = BertTokenizer.from_pretrained(config.get("model", "bert_path"))
        self.k = config.getint("data", "topk")
        # if mode in ['valid','test']:
        #     self.siglemulti = SingleMulti('gbt/statement_tfidf.model', 'gbt/statement_som_gbt.model')

    def convert(self, tokens, which, l):
        mask = []
        tokenx = []

        tokens = self.tokenizer.tokenize(tokens)
        ids = self.tokenizer.convert_tokens_to_ids(tokens)

        for a in range(0, len(ids)):
            mask.append(1)
            tokenx.append(which)

        while len(ids) < l:
            ids.append(self.tokenizer.vocab["[PAD]"])
            mask.append(0)
            tokenx.append(which)

        ids = torch.LongTensor(ids)
        mask = torch.LongTensor(mask)
        tokenx = torch.LongTensor(tokenx)

        return ids, mask, tokenx

    def convertx(self, token_list, max_seq_len):
        all_input_ids, all_input_mask, all_segment_ids = [], [], []

        input_ids = self.tokenizer.convert_tokens_to_ids(['[CLS]'])
        input_masks = [1]
        segment_ids = [0]
        # all_input_ids.append(input_ids)
        # all_input_mask.append(input_masks)
        # all_segment_ids.append(segment_ids)

        for i, _ in enumerate(token_list):
            tokens = self.tokenizer.tokenize(token_list[i]) + ['[SEP]']
            input_ids += self.tokenizer.convert_tokens_to_ids(tokens)
            input_masks += [1]*(len(tokens))
            segment_ids += [i]*(len(tokens))

        tokens_len = len(input_ids)
        if tokens_len > 512:
            input_ids = input_ids[:256] + input_ids[-256:]
            input_masks = input_masks[:256] + input_masks[-256:]
            segment_ids = segment_ids[:256] + segment_ids[-256:]
        else:
            input_ids += [0] * (512 - tokens_len)
            segment_ids += [0] * (512 - tokens_len)
            input_masks += [0] * (512 - tokens_len)

        assert len(input_ids) == 512, len(input_ids)
        assert len(input_masks) == 512,len(input_masks)
        assert len(segment_ids) == 512,len(segment_ids)

        all_input_ids.append(input_ids)
        all_input_mask.append(input_masks)
        all_segment_ids.append(segment_ids)

        all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
        all_input_mask = torch.tensor(all_input_mask, dtype=torch.long)
        all_segment_ids = torch.tensor(all_segment_ids, dtype=torch.long)

        return all_input_ids, all_input_mask, all_segment_ids

    def process(self, data, config, mode, *args, **params):
        txt = []
        mask = []
        token = []
        label = []
        idx = []
        sm = []

        for temp_data in data:
            idx.append(temp_data["id"])

            # if mode in ['train', 'valid']:
            #     # clean up answers.
            temp_data["answer"] = [a for a in temp_data["answer"] if a != "。"]
            #     sm.append(len(temp_data["answer"])>1)
            # else:
            sm.append(1)#多选
            #     states = temp_data["statement"]
            #     for option in ["A", "B", "C", "D"]:
            #         states += temp_data["option_list"][option]
            #     isms = self.siglemulti.checkSingleMulti(states)
            #     sm.append(isms)

            label_x=0
            if set(temp_data["answer"]) == set(['A']):
                label_x = 0
            if set(temp_data["answer"]) == set(['B']):
                label_x = 1
            if set(temp_data["answer"]) == set(['C']):
                label_x = 2
            if set(temp_data["answer"]) == set(['D']):
                label_x = 3
            if set(temp_data["answer"]) == set(['A', 'B']):
                label_x = 4
            if set(temp_data["answer"]) == set(['A', 'C']):
                label_x = 5
            if set(temp_data["answer"]) == set(['B', 'C']):
                label_x = 6
            if set(temp_data["answer"]) == set(['A', 'B', 'C']):
                label_x = 7
            if set(temp_data["answer"]) == set(['A', 'D']):
                label_x = 8
            if set(temp_data["answer"]) == set(['B', 'D']):
                label_x = 9
            if set(temp_data["answer"]) == set(['A', 'B', 'D']):
                label_x = 10
            if set(temp_data["answer"]) == set(['C', 'D']):
                label_x = 11
            if set(temp_data["answer"]) == set(['A', 'C', 'D']):
                label_x = 12
            if set(temp_data["answer"]) == set(['B', 'C', 'D']):
                label_x = 13
            if set(temp_data["answer"]) == set(['A', 'B', 'C', 'D']):
                label_x = 14
            label.append(label_x)

            # temp_text = []
            # temp_mask = []
            # temp_token = []

            text = temp_data["statement"][:self.max_len1]
            textlinst = [text] + [temp_data["option_list"][option] for option in ["A", "B", "C", "D"]]
            txt1, mask1, token1 = self.convertx(textlinst, self.max_len1)

            txt.append(txt1)
            mask.append(mask1)
            token.append(token1)

            # txt1, mask1, token1 = self.convert(text, 0, self.max_len1)
            # temp_text.append(torch.cat([txt1]))
            # temp_mask.append(torch.cat([mask1]))
            # temp_token.append(torch.cat([token1]))
            #
            # for option in ["A", "B", "C", "D"]:
            #     text = temp_data["option_list"][option]
            #     txt1, mask1, token1 = self.convert(text, ord(option)-ord('A') + 1, self.max_len2)
            #     temp_text.append(torch.cat([txt1]))
            #     temp_mask.append(torch.cat([mask1]))
            #     temp_token.append(torch.cat([token1]))


            # txt.append(torch.stack(temp_text))
            # mask.append(torch.stack(temp_mask))
            # token.append(torch.stack(temp_token))

        txt = torch.stack(txt)
        mask = torch.stack(mask)
        token = torch.stack(token)
        label = torch.LongTensor(np.array(label, dtype=np.int32))

        return {"text": txt, "mask": mask, "token": token, 'label': label, "id": idx, "sorm": sm}

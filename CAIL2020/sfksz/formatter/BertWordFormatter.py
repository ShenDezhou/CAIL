import json
import torch
import numpy as np
import os
#from pytorch_pretrained_bert import BertTokenizer
from transformers import BertTokenizer

class BertWordFormatter:
    def __init__(self, config, mode):
        self.max_question_len = config.getint("data", "max_question_len")
        self.max_option_len = config.getint("data", "max_option_len")

        self.tokenizer = BertTokenizer.from_pretrained(config.get("model", "bert_path"))

    def convert_tokens_to_ids(self, tokens):
        arr = []
        for a in range(0, len(tokens)):
            if tokens[a] in self.word2id:
                arr.append(self.word2id[tokens[a]])
            else:
                arr.append(self.word2id["UNK"])
        return arr

    def convert(self, tokens, l, bk=False):
        tokens = "".join(tokens)
        # while len(tokens) < l:
        #     tokens.append("PAD")
        # if bk:
        #     tokens = tokens[len(tokens) - l:]
        # else:
        #     tokens = tokens[:l]
        ids = self.tokenizer.tokenize(tokens)

        return ids

    def _convert_sentence_pair_to_bert_dataset(
            self, context, max_len):
        """Convert sentence pairs to dataset for BERT model.

        Args:
            sc_list, bc_list: List[List[str]], list of word tokens list
            label_list: train: List[int], list of labels
                        test: []

        Returns:
            Train:
                torch.utils.data.TensorDataset
                    each record: (input_ids, input_mask, segment_ids, label)
            Test:
                torch.utils.data.TensorDataset
                    each record: (input_ids, input_mask, segment_ids)
        """
        all_input_ids, all_input_mask, all_segment_ids = [], [], []
        for i, _ in enumerate(context):
            tokens = ['[CLS]'] + context[i] + ['[SEP]']
            segment_ids = [0] * len(tokens)
            if len(tokens) > max_len:
                tokens = tokens[-max_len:]
                assert len(tokens) == max_len
                segment_ids = segment_ids[-max_len:]
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)
            tokens_len = len(input_ids)
            input_ids += [0] * (max_len - tokens_len)
            segment_ids += [0] * (max_len - tokens_len)
            input_mask += [0] * (max_len - tokens_len)
            all_input_ids.append(input_ids)
            all_input_mask.append(input_mask)
            all_segment_ids.append(segment_ids)
        all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
        all_input_mask = torch.tensor(all_input_mask, dtype=torch.long)
        all_segment_ids = torch.tensor(all_segment_ids, dtype=torch.long)
        # test
        return (
            all_input_ids, all_input_mask, all_segment_ids)

    def process(self, data, config, mode, *args, **params):
        context = []
        question = []
        label = []
        idx = []

        for temp_data in data:
            idx.append(temp_data["id"])

            if mode != "test":
                # label_x = []
                # for opt in list("ABCD"):
                #     if opt in temp_data["answer"]:
                #         label_x.append(1)
                #     else:
                #         label_x.append(0)

                label_x = -1
                if "A" in temp_data["answer"]:
                    label_x += 1
                if "B" in temp_data["answer"]:
                    label_x += 2
                if "C" in temp_data["answer"]:
                    label_x += 4
                if "D" in temp_data["answer"]:
                    label_x += 8
                label.append(label_x)

            temp_context = []
            temp_question = []

            temp_question.append(self.convert(temp_data["statement"], self.max_question_len, bk=True))
            for option in ["A", "B", "C", "D"]:
                temp_context.append(self.convert(temp_data["option_list"][option], self.max_option_len))

            context.extend(temp_context)
            question.extend(temp_question)

        # question = torch.tensor(question, dtype=torch.long)
        # context = torch.tensor(context, dtype=torch.long)
        question = self._convert_sentence_pair_to_bert_dataset(question, self.max_question_len)
        context = self._convert_sentence_pair_to_bert_dataset(context, self.max_option_len)
        if mode != "test":
            label = torch.LongTensor(np.array(label, dtype=np.int))
            return {"context": context, "question": question, 'label': label, "id": idx}
        else:
            return {"context": context, "question": question, "id": idx}

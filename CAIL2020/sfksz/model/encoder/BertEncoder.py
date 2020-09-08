import torch
import torch.nn as nn
import torch.nn.functional as F
# from pytorch_pretrained_bert import BertModel
from tqdm import tqdm
from transformers import BertTokenizer
from transformers.modeling_bert import BertModel

class BertEncoder(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(BertEncoder, self).__init__()
        self.bert = BertModel.from_pretrained(config.get("model", "bert_path"))
        self.tokenizer = BertTokenizer.from_pretrained(config.get("model", "bert_path"))
        for param in self.bert.parameters():
            param.requires_grad = True

    def forward(self, s1_list, max_seq_len):
        all_input_ids, all_input_mask, all_segment_ids = [], [], []
        for i, _ in enumerate(s1_list):
            tokens = ['[CLS]']
            segment_ids = [0]
            for j in range(len(s1_list[i])):
                token_by_char = list("".join(s1_list[i][j]))
                tokens += token_by_char + ['[SEP]']
                segment_ids += [j] * (len(token_by_char) + 1)
            if len(tokens) > max_seq_len:
                tokens = tokens[:max_seq_len]
                segment_ids = segment_ids[:max_seq_len]
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)
            tokens_len = len(input_ids)
            input_ids += [0] * (max_seq_len - tokens_len)
            segment_ids += [0] * (max_seq_len - tokens_len)
            input_mask += [0] * (max_seq_len - tokens_len)
            assert len(input_ids) == max_seq_len, i
            assert len(segment_ids) == max_seq_len, i
            assert len(input_mask) == max_seq_len, i
            all_input_ids.append(input_ids)
            all_input_mask.append(input_mask)
            all_segment_ids.append(segment_ids)
        all_input_ids = torch.tensor(all_input_ids, dtype=torch.long).cuda()
        all_input_mask = torch.tensor(all_input_mask, dtype=torch.long).cuda()
        all_segment_ids = torch.tensor(all_segment_ids, dtype=torch.long).cuda()

        y = self.bert(input_ids=all_input_ids,attention_mask=all_input_mask,token_type_ids=all_segment_ids,
                        output_hidden_states=False)[0]

        return y

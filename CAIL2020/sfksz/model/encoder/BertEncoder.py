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
        self.max_seq_len =config.getint("model", "max_seq_len")
        for param in self.bert.parameters():
            param.requires_grad = True

    def forward(self, all_input_ids, all_input_mask, all_segment_ids):
        if torch.cuda.is_available():
            all_input_ids = all_input_ids.cuda()
            all_input_mask = all_input_mask.cuda()
            all_segment_ids = all_segment_ids.cuda()

        return self.bert(input_ids=all_input_ids,attention_mask=all_input_mask,token_type_ids=all_segment_ids,
                        output_hidden_states=True)

import argparse
import json
import os
from types import SimpleNamespace

import fire
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data import Data
from evaluate import evaluate
from model import BertSupportNetX
from utils import load_torch_model
from utils import get_path,get_csv_logger


from os.path import join
from tqdm import tqdm
import json
import torch
from model import *
from tools.utils import convert_to_tokens
import numpy as np
import random


if __name__ == "__main__":
    config_file = 'config/bert_config.json'
    # 0. Load config and mkdir
    with open(config_file) as fin:
        config = json.load(fin, object_hook=lambda d: SimpleNamespace(**d))
    get_path(os.path.join(config.model_path, config.experiment_name))
    get_path(config.log_path)
    get_path(config.prediction_path)

    if config.model_type == 'rnn':  # build vocab for rnn
        build_vocab(file_in=config.all_train_file_path,
                    file_out=os.path.join(config.model_path, 'vocab.txt'))
    # 1. Load data
    data = Data(vocab_file=os.path.join(config.model_path, 'vocab.txt'),
                max_seq_len=config.max_seq_len,
                model_type=config.model_type, config=config)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    exam, feats, dataset = data.load_file(config.test_file_path, True)

    data_loader = DataLoader(dataset, batch_size=config.batch_size)
    # 2. Build model
    model = BertSupportNetX(config)
    #load model states.
    if config.trained_weight:
        model.load_state_dict(torch.load(config.trained_weight))
    model.to(device)

    model.eval()
    answer_dict = {}
    sp_dict = {}
    # dataloader.refresh()
    total_test_loss = [0] * 5
    # context_idxs, context_mask, segment_idxs,
    # query_mapping, all_mapping,
    # ids, y1, y2, q_type,
    # start_mapping,
    # is_support
    tqdm_obj = tqdm(data_loader, ncols=80)
    for step, batch in enumerate(tqdm_obj):
        batch = tuple(t.to(device) for t in batch)
        # batch['context_mask'] = batch['context_mask'].float()
        start_logits, end_logits, type_logits, sp_logits, start_position, end_position = model(*batch)
        # loss1 = self.criterion(start_logits, batch[6]) + self.criterion(end_logits, batch[7])  # y1,y2
        # loss2 = self.config.type_lambda * self.criterion(type_logits, batch[8])  # q_type
        # # sp_value = self.sp_loss_fct(sp_logits.view(-1), batch[10].float().view(-1)).sum()
        # # loss3 = self.config.sp_lambda * sp_value / batch[9].sum()
        #
        # loss = loss1 + loss2
        # loss_list = [loss, loss1, loss2]
        #
        # for i, l in enumerate(loss_list):
        #     if not isinstance(l, int):
        #         total_test_loss[i] += l.item()

        batchsize = batch[0].size(0)
        # ids
        answer_dict_ = convert_to_tokens(exam, feats, batch[5], start_position.data.cpu().numpy().tolist(),
                                         end_position.data.cpu().numpy().tolist(),
                                         np.argmax(type_logits.data.cpu().numpy(), 1))
        answer_dict.update(answer_dict_)

        # predict_support_np = torch.sigmoid(sp_logits).data.cpu().numpy()
        # for i in range(predict_support_np.shape[0]):
        #     cur_sp_pred = []
        #     cur_id = batch[5][i].item()
        #
        #     cur_sp_logit_pred = []  # for sp logit output
        #     for j in range(predict_support_np.shape[1]):
        #         if j >= len(exam[cur_id].sent_names):
        #             break
        #         if need_sp_logit_file:
        #             temp_title, temp_id = exam[cur_id].sent_names[j]
        #             cur_sp_logit_pred.append((temp_title, temp_id, predict_support_np[i, j]))
        #         if predict_support_np[i, j] > self.config.sp_threshold:
        #             cur_sp_pred.append(exam[cur_id].sent_names[j])
        #     sp_dict.update({cur_id: cur_sp_pred})

    new_answer_dict = {}
    for key, value in answer_dict.items():
        new_answer_dict[key] = value.replace(" ", "")
    prediction = {'answer': new_answer_dict, 'sp': sp_dict}
    with open(join(config.prediction_path, 'pred.json'), 'w', encoding='utf8') as f:
        json.dump(prediction, f, indent=4, ensure_ascii=False)


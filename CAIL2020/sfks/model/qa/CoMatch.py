'''
Copyright 2015 Singapore Management University (SMU). All Rights Reserved.
Permission to use, copy, modify and distribute this software and its documentation for purposes of research, teaching and general academic pursuits, without fee and without a signed licensing agreement, is hereby granted, provided that the above copyright statement, this paragraph and the following paragraph on disclaimer appear in all copies, modifications, and distributions.  Contact Singapore Management University, Intellectual Property Management Office at iie@smu.edu.sg, for commercial licensing opportunities.
This software is provided by the copyright holder and creator "as is" and any express or implied warranties, including, but not Limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed.  In no event shall SMU or the creator be liable for any direct, indirect, incidental, special, exemplary or consequential damages, however caused arising in any way out of the use of this software.
'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

import json
from tools.accuracy_tool import single_label_top1_accuracy


def masked_softmax(vector, seq_lens):
    mask = vector.new(vector.size()).zero_()
    for i in range(seq_lens.size(0)):
        mask[i, :, :seq_lens[i]] = 1
    mask = Variable(mask, requires_grad=False)

    if mask is None:
        result = torch.nn.functional.softmax(vector, dim=-1)
    else:
        result = torch.nn.functional.softmax(vector * mask, dim=-1)
        result = result * mask
        result = result / (result.sum(dim=-1, keepdim=True) + 1e-13)
    return result


class MatchNet(nn.Module):
    def __init__(self, mem_dim, dropoutP):
        super(MatchNet, self).__init__()
        self.map_linear = nn.Linear(2 * mem_dim, 2 * mem_dim)
        self.trans_linear = nn.Linear(mem_dim, mem_dim)
        self.drop_module = nn.Dropout(dropoutP)

    def forward(self, inputs):
        proj_p, proj_q, seq_len = inputs
        trans_q = self.trans_linear(proj_q)
        att_weights = proj_p.bmm(torch.transpose(proj_q, 1, 2))
        att_norm = masked_softmax(att_weights, seq_len)

        att_vec = att_norm.bmm(proj_q)
        elem_min = att_vec - proj_p
        elem_mul = att_vec * proj_p
        all_con = torch.cat([elem_min, elem_mul], 2)
        output = nn.ReLU()(self.map_linear(all_con))
        return output


class MaskLSTM(nn.Module):
    def __init__(self, in_dim, out_dim, layers=1, batch_first=True, bidirectional=True, dropoutP=0.3):
        super(MaskLSTM, self).__init__()
        self.lstm_module = nn.LSTM(in_dim, out_dim, layers, batch_first=batch_first, bidirectional=bidirectional,
                                   dropout=dropoutP)
        self.drop_module = nn.Dropout(dropoutP)

    def forward(self, inputs):
        input, seq_lens = inputs
        mask_in = input.new(input.size()).zero_()
        for i in range(seq_lens.size(0)):
            mask_in[i, :seq_lens[i]] = 1
        mask_in = Variable(mask_in, requires_grad=False)

        input_drop = self.drop_module(input * mask_in)

        H, _ = self.lstm_module(input_drop)

        mask = H.new(H.size()).zero_()
        for i in range(seq_lens.size(0)):
            mask[i, :seq_lens[i]] = 1
        mask = Variable(mask, requires_grad=False)

        output = H * mask

        return output


class CoMatch(nn.Module):
    def __init__(self, config):
        super(CoMatch, self).__init__()
        self.emb_dim = config.getint("model", "hidden_size")  # 300
        self.mem_dim = config.getint("model", "hidden_size")  # 150
        self.dropoutP = config.getfloat("model", "dropout")  # args.dropoutP 0.2
        # self.cuda_bool = args.cuda

        self.word_num = len(json.load(open(config.get("data", "word2id"), "r")))

        self.embs = nn.Embedding(self.word_num, self.emb_dim)

        self.encoder = MaskLSTM(self.emb_dim, self.mem_dim, dropoutP=self.dropoutP)
        self.l_encoder = MaskLSTM(self.mem_dim * 8, self.mem_dim, dropoutP=self.dropoutP)
        self.h_encoder = MaskLSTM(self.mem_dim * 2, self.mem_dim, dropoutP=0)

        self.match_module = MatchNet(self.mem_dim * 2, self.dropoutP)

        self.rank_module = nn.Linear(self.mem_dim * 2 * config.getint("data", "topk"), 1)

        self.multi = config.getboolean("data", "multi_choice")
        self.multi_module = nn.Linear(4, 16)

        self.drop_module = nn.Dropout(self.dropoutP)

    def init_multi_gpu(self, device, config, *args, **params):
        self.embs = nn.DataParallel(self.embs)
        self.encoder = nn.DataParallel(self.encoder)
        self.l_encoder = nn.DataParallel(self.l_encoder)
        self.h_encoder = nn.DataParallel(self.h_encoder)
        self.match_module = nn.DataParallel(self.match_module)
        self.rank_module = nn.DataParallel(self.rank_module)

    def forward(self, inputs):
        documents, questions, options = inputs
        d_word, d_h_len, d_l_len = documents
        o_word, o_h_len, o_l_len = options
        q_word, q_len = questions
        # print("d_word", d_word.size())
        # print("d_h_len", d_h_len.size())
        # print("d_l_len", d_l_len.size())
        # print("o_word", o_word.size())
        # print("o_h_len", o_h_len.size())
        # print("o_l_len", o_l_len.size())
        # print("q_word", q_word.size())
        # print("q_len", q_len.size())

        batch = d_word.size()[0]
        option = d_word.size()[1]
        k = d_word.size()[2]

        d_embs = self.drop_module(self.embs(d_word))
        d_embs = torch.zeros(d_embs.shape).cuda()
        o_embs = self.drop_module(self.embs(o_word))
        q_embs = self.drop_module(self.embs(q_word))
        # print("d_embs", d_embs.size())
        # print("o_embs", o_embs.size())
        # print("q_embs", q_embs.size())

        d_hidden = self.encoder(
            [d_embs.view(d_embs.size(0) * d_embs.size(1) * d_embs.size(2) * d_embs.size(3), d_embs.size(4),
                         self.emb_dim),
             d_l_len.view(-1)])
        o_hidden = self.encoder(
            [o_embs.view(o_embs.size(0) * o_embs.size(1), o_embs.size(2), self.emb_dim), o_l_len.view(-1)])
        q_hidden = self.encoder([q_embs, q_len])

        # print("d_hidden", d_hidden.size())
        # print("o_hidden", o_hidden.size())
        # print("q_hidden", q_hidden.size())

        # d_hidden_3d = d_hidden.view(d_embs.size(0), d_embs.size(1) * d_embs.size(2), d_hidden.size(-1))
        # d_hidden_3d_repeat = d_hidden_3d.repeat(1, o_embs.size(1), 1).view(d_hidden_3d.size(0) * o_embs.size(1),
        #                                                                   d_hidden_3d.size(1), d_hidden_3d.size(2))
        d_hidden_3d_repeat = d_hidden.view(d_word.size()[0] * d_word.size()[2] * o_embs.size(1), -1,
                                           d_hidden.size()[-1])

        # print("d_hidden_3d", d_hidden_3d.size())
        # print("d_hidden_3d_repeat", d_hidden_3d_repeat.size())

        q_hidden_repeat = q_hidden.repeat(1, o_embs.size(1) * d_word.size()[2], 1).view(
            q_hidden.size()[0] * o_embs.size(1) * d_word.size()[2], q_hidden.size()[1], q_hidden.size()[2])
        q_len_repeat = q_len.repeat(o_embs.size(1) * d_word.size()[2])
        # print("q_hidden_repeat", q_hidden_repeat.size())
        # print("q_len_repeat", q_len_repeat.size())

        o_hidden_repeat = o_hidden.repeat(1, 1, d_word.size()[2], ).view(o_hidden.size()[0] * d_word.size()[2],
                                                                         o_hidden.size()[1], o_hidden.size()[2])
        o_l_len_repeat = o_l_len.repeat(1, d_word.size()[2]).view(o_l_len.size()[0] * d_word.size()[2],
                                                                  o_l_len.size()[1])
        # print("o_hidden_repeat", o_hidden_repeat.size())
        # print("o_l_len_repeat", o_l_len_repeat.size())

        do_match = self.match_module([d_hidden_3d_repeat, o_hidden_repeat, o_l_len_repeat.view(-1)])
        dq_match = self.match_module([d_hidden_3d_repeat, q_hidden_repeat, q_len_repeat])

        # print("do_match", do_match.size())
        # print("dq_match", dq_match.size())

        dq_match_repeat = dq_match
        # print("dq_match_repeat", dq_match_repeat.size())

        co_match = torch.cat([do_match, dq_match_repeat], -1)

        # print("co_match", co_match.size())

        co_match_hier = co_match.view(d_embs.size(0) * o_embs.size(1) * d_embs.size(2) * d_embs.size(3), d_embs.size(4),
                                      -1)
        # print("co_match_hier", co_match_hier.size())

        l_hidden = self.l_encoder([co_match_hier, d_l_len.view(-1)])
        # print("l_hidden", l_hidden.size())
        l_hidden_pool, _ = l_hidden.max(1)
        # print("l_hidden_pool", l_hidden_pool.size())

        h_hidden = self.h_encoder(
            [l_hidden_pool.view(d_embs.size(0) * o_embs.size(1) * d_embs.size(2), d_embs.size(3), -1),
             d_h_len.view(-1, 1).view(-1)])
        # print("h_hidden", h_hidden.size())
        h_hidden_pool, _ = h_hidden.max(1)
        # print("h_hidden_pool", h_hidden_pool.size())

        # o_rep = h_hidden_pool.view(d_embs.size(0), o_embs.size(1), -1)
        # print("o_rep", o_rep.size())
        # output = self.rank_module(o_rep).squeeze(2)

        o_rep = h_hidden_pool.view(d_embs.size(0), o_embs.size(1), -1)
        output = self.rank_module(o_rep).squeeze(2)

        if self.multi:
            output = self.multi_module(output)

        return output


class CoMatching(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(CoMatching, self).__init__()

        self.co_match = CoMatch(config)

        self.criterion = nn.CrossEntropyLoss()
        self.accuracy_function = single_label_top1_accuracy

    def init_multi_gpu(self, device, config, *args, **params):
        self.co_match.init_multi_gpu(device, config, *args, **params)

    def forward(self, data, config, gpu_list, acc_result, mode):
        q, ql = data["question"], data["question_len"]
        o, oh, ol = data["option"], data["option_sent"], data["option_len"]
        d, dh, dl = data["document"], data["document_sent"], data["document_len"]
        label = data["label"]

        x = [[d, dh, dl], [q, ql], [o, oh, ol]]
        y = self.co_match(x)

        loss = self.criterion(y, label)
        acc_result = self.accuracy_function(y, label, config, acc_result)
        return {"loss": loss, "acc_result": acc_result}

import torch
import numpy as np
from numpy.random import shuffle

IGNORE_INDEX = -100


class DataIteratorPack(object):
    def __init__(self, features, example_dict,bsz, device, sent_limit, entity_limit,
                 entity_type_dict=None, sequential=False,):
        self.bsz = bsz   # batch_size
        self.device = device
        self.features = features
        self.example_dict = example_dict
        # self.entity_type_dict = entity_type_dict
        self.sequential = sequential
        self.sent_limit = sent_limit
        # self.para_limit = 4  # 默认值，有必要再重设
        # self.entity_limit = entity_limit
        self.example_ptr = 0
        if not sequential:
            shuffle(self.features)  # 只shuffle feature，别的还能对得上吗？

    def refresh(self):
        self.example_ptr = 0
        if not self.sequential:
            shuffle(self.features)

    def empty(self):
        return self.example_ptr >= len(self.features)

    def __len__(self):
        return int(np.ceil(len(self.features)/self.bsz))

    def __iter__(self):
        # BERT input
        context_idxs = torch.LongTensor(self.bsz, 512)
        context_mask = torch.LongTensor(self.bsz, 512)
        segment_idxs = torch.LongTensor(self.bsz, 512)

        # Graph and Mappings   注意这些是在gpu里的，可能是因为训练的过程中要用到

        query_mapping = torch.Tensor(self.bsz, 512).cuda(self.device)
        start_mapping = torch.Tensor(self.bsz, self.sent_limit, 512).cuda(self.device)
        all_mapping = torch.Tensor(self.bsz, 512, self.sent_limit).cuda(self.device)


        # Label tensor
        y1 = torch.LongTensor(self.bsz).cuda(self.device)   # 之前不是一个answer对应好几个span吗？
        y2 = torch.LongTensor(self.bsz).cuda(self.device)
        q_type = torch.LongTensor(self.bsz).cuda(self.device)   # 这个应该是answer_type而不是question type
        is_support = torch.FloatTensor(self.bsz, self.sent_limit).cuda(self.device)


        # bfs_mask = torch.FloatTensor(self.bsz, self.n_layers, self.entity_limit).cuda(self.device)  # (batch, 2, 80)

        while True:
            if self.example_ptr >= len(self.features):
                break
            start_id = self.example_ptr  # 这个example_ptr是干啥用的
            cur_bsz = min(self.bsz, len(self.features) - start_id)   # 用来处理剩余样本不足一个batch的情况
            cur_batch = self.features[start_id: start_id + cur_bsz]  # 一个batch大小的feature
            cur_batch.sort(key=lambda x: sum(x.doc_input_mask), reverse=True)  # 输入长的在前面？有啥用？

            ids = []
            max_sent_cnt = 0
            max_entity_cnt = 0
            for mapping in [start_mapping, all_mapping,  query_mapping]:
                mapping.zero_()   # 把几个mapping都初始化为0
            # 为啥要用-100填充
            is_support.fill_(0)
            # is_support.fill_(0)  # BCE

            for i in range(len(cur_batch)):    # 遍历当前batch，把每个样本的bert输入填进去
                case = cur_batch[i]            # 一个feature
                # print(f'all_doc_tokens is {case.doc_tokens}')
                context_idxs[i].copy_(torch.Tensor(case.doc_input_ids))
                context_mask[i].copy_(torch.Tensor(case.doc_input_mask))
                segment_idxs[i].copy_(torch.Tensor(case.doc_segment_ids))
                # print(case)
                # print(case.sent_spans)
                # query 对应的token位置为1
                for j in range(case.sent_spans[0][0] - 1):
                    query_mapping[i, j] = 1

                # adj = torch.from_numpy(tem_graph['adj'])
                # start_entities = torch.from_numpy(tem_graph['start_entities'])
                # for l in range(self.n_layers):
                #     bfs_mask[i][l].copy_(start_entities)  # 每一层的mask都是实体的起点
                #     start_entities = bfs_step(start_entities, adj)  # 返回第l步能到达的实体，l为0和1

                if case.ans_type == 0:
                    if len(case.end_position) == 0:
                        y1[i] = y2[i] = 0   # 如果结束位置是0，span的标签就为0
                    elif case.end_position[0] < 512:
                        y1[i] = case.start_position[0]   # 只用第一个找到的span
                        y2[i] = case.end_position[0]
                    else:
                        y1[i] = y2[i] = 0
                    q_type[i] = 0
                elif case.ans_type == 1:
                    y1[i] = IGNORE_INDEX  # span是-100
                    y2[i] = IGNORE_INDEX
                    q_type[i] = 1  # 这个明明是answer_type，非要叫q_type
                elif case.ans_type == 2:
                    y1[i] = IGNORE_INDEX
                    y2[i] = IGNORE_INDEX
                    q_type[i] = 2
                elif case.ans_type == 3:
                    y1[i] = IGNORE_INDEX
                    y2[i] = IGNORE_INDEX
                    q_type[i] = 3

                for j, sent_span in enumerate(case.sent_spans[:self.sent_limit]):   # 句子序号，span
                    is_sp_flag = j in case.sup_fact_ids   # 这个代码写的真几把烂#我也觉得
                    start, end = sent_span
                    if start < end:  # 还有start大于end的时候？
                        is_support[i, j] = int(is_sp_flag)   # 样本i的第j个句子是否是sp
                        all_mapping[i, start:end+1, j] = 1   # （batch_size, 512, 20) 第j个句子开始和结束全为1
                        start_mapping[i, j, start] = 1       # （batch_size, 20, 512)




                ids.append(case.qas_id)
                max_sent_cnt = max(max_sent_cnt, len(case.sent_spans))


            input_lengths = (context_mask[:cur_bsz] > 0).long().sum(dim=1)
            max_c_len = int(input_lengths.max())

            self.example_ptr += cur_bsz

            yield {
                'context_idxs': context_idxs[:cur_bsz, :max_c_len].contiguous(),#used
                'context_mask': context_mask[:cur_bsz, :max_c_len].contiguous(),#used
                'segment_idxs': segment_idxs[:cur_bsz, :max_c_len].contiguous(),#used
                'query_mapping': query_mapping[:cur_bsz, :max_c_len].contiguous(),#used
                'y1': y1[:cur_bsz],#uesed
                'y2': y2[:cur_bsz],#uesed
                'ids': ids,#used
                'q_type': q_type[:cur_bsz],#uesed
                'start_mapping': start_mapping[:cur_bsz, :max_sent_cnt, :max_c_len],#used
                'all_mapping': all_mapping[:cur_bsz, :max_c_len, :max_sent_cnt],#used
                # 'bfs_mask': bfs_mask[:cur_bsz, :, :max_entity_cnt],
                'is_support':is_support[:cur_bsz, :max_sent_cnt].contiguous(),
            }

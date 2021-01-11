"""Data processor for SMP-CAIL2020-Argmine.

Author: Tsinghuaboy (tsinghua9boy@sina.com)

In data file, each line contains 1 sc sentence and 5 bc sentences.
The data processor convert each line into 5 samples,
each sample with 1 sc sentence and 1 bc sentence.

Usage:
1. Tokenizer (used for RNN model):
    from data import Tokenizer
    vocab_file = 'vocab.txt'
    sentence = '我饿了，想吃东西了。'
    tokenizer = Tokenizer(vocab_file)
    tokens = tokenizer.tokenize(sentence)
    # ['我', '饿', '了', '，', '想', '吃', '东西', '了', '。']
    ids = tokenizer.convert_tokens_to_ids(tokens)
2. Data:
    from data import Data
    # For training, load train and valid set
    # For BERT model
    data = Data('model/bert/vocab.txt', model_type='bert')
    datasets = data.load_train_and_valid_files(
        'SMP-CAIL2020-train.csv', 'SMP-CAIL2020-valid.csv')
    train_set, valid_set_train, valid_set_valid = datasets
    # For RNN model
    data = Data('model/rnn/vocab.txt', model_type='rnn')
    datasets = data.load_all_files(
        'SMP-CAIL2020-train.csv', 'SMP-CAIL2020-valid.csv')
    train_set, valid_set_train, valid_set_valid = datasets
    # For testing, load test set
    data = Data('model/bert/vocab.txt', model_type='bert')
    test_set = data.load_file('SMP-CAIL2020-test.csv', train=False)
"""
import json
from typing import List
import jieba
import torch

import pandas as pd

from torch.utils.data import TensorDataset
from transformers import BertTokenizer
# from pytorch_pretrained_bert import BertTokenizer
from tqdm import tqdm

max_support_sents = 26

class Tokenizer:
    """Tokenizer for Chinese given vocab.txt.

    Attributes:
        dictionary: Dict[str, int], {<word>: <index>}
    """
    def __init__(self, vocab_file='vocab.txt'):
        """Initialize and build dictionary.

        Args:
            vocab_file: one word each line
        """
        self.dictionary = {'[PAD]': 0, '[UNK]': 1}
        count = 2
        with open(vocab_file, encoding='utf-8') as fin:
            for line in fin:
                word = line.strip()
                self.dictionary[word] = count
                count += 1

    def __len__(self):
        return len(self.dictionary)

    @staticmethod
    def tokenize(sentence: str) -> List[str]:
        """Cut words for a sentence.

        Args:
            sentence: sentence

        Returns:
            words list
        """
        return jieba.lcut(sentence)

    def convert_tokens_to_ids(
            self, tokens_list: List[str]) -> List[int]:
        """Convert tokens to ids.

        Args:
            tokens_list: word list

        Returns:
            index list
        """
        return [self.dictionary.get(w, 1) for w in tokens_list]


class Example(object):

    def __init__(self,
                 qas_id,
                 qas_type,
                 doc_tokens,
                 question_text,
                 sent_num,
                 sent_names,
                 sup_fact_id,
                 para_start_end_position,
                 sent_start_end_position,
                 entity_start_end_position,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None):
        self.qas_id = qas_id
        self.qas_type = qas_type
        self.doc_tokens = doc_tokens
        self.question_text = question_text
        self.sent_num = sent_num
        self.sent_names = sent_names
        self.sup_fact_id = sup_fact_id
        self.para_start_end_position = para_start_end_position
        self.sent_start_end_position = sent_start_end_position
        self.entity_start_end_position = entity_start_end_position
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 qas_id,
                 doc_tokens,
                 doc_input_ids,
                 doc_input_mask,
                 doc_segment_ids,
                 query_tokens,
                 query_input_ids,
                 query_input_mask,
                 query_segment_ids,
                 sent_spans,
                 sup_fact_ids,
                 ans_type,
                 token_to_orig_map,
                 start_position=None,
                 end_position=None):

        self.qas_id = qas_id
        self.doc_tokens = doc_tokens
        self.doc_input_ids = doc_input_ids
        self.doc_input_mask = doc_input_mask
        self.doc_segment_ids = doc_segment_ids

        self.query_tokens = query_tokens
        self.query_input_ids = query_input_ids
        self.query_input_mask = query_input_mask
        self.query_segment_ids = query_segment_ids

        self.sent_spans = sent_spans
        self.sup_fact_ids = sup_fact_ids
        self.ans_type = ans_type
        self.token_to_orig_map=token_to_orig_map

        self.start_position = start_position
        self.end_position = end_position


def get_valid_spans(spans, limit):
    new_spans = []
    for span in spans:
        if span[1] < limit:
            new_spans.append(span)
        else:
            new_span = list(span)
            new_span[1] = limit - 1
            new_spans.append(tuple(new_span))
            break
    return new_spans


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return new_start, new_end

    return input_start, input_end


class Data:
    """Data processor for BERT and RNN model for SMP-CAIL2020-Argmine.

    Attributes:
        model_type: 'bert' or 'rnn'
        max_seq_len: int, default: 512
        tokenizer:  BertTokenizer for bert
                    Tokenizer for rnn
    """
    def __init__(self,
                 vocab_file='',
                 max_seq_len: int = 512,
                 model_type: str = 'bert', config=None):
        """Initialize data processor for SMP-CAIL2020-Argmine.

        Args:
            vocab_file: one word each line
            max_seq_len: max sequence length, default: 512
            model_type: 'bert' or 'rnn'
                If model_type == 'bert', use BertTokenizer as tokenizer
                Otherwise, use Tokenizer as tokenizer
        """
        self.model_type = model_type
        if self.model_type == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained(config.bert_model_path)#BertTokenizer(vocab_file)
        else:  # rnn
            self.tokenizer = Tokenizer(vocab_file)
        self.max_seq_len = max_seq_len

    def load_file(self,
                  file_path='SMP-CAIL2020-train.csv',
                  train=True) -> TensorDataset:
        """Load SMP-CAIL2020-Argmine train file and construct TensorDataset.

        Args:
            file_path: train file with last column as label
            train:
                If True, train file with last column as label
                Otherwise, test file without last column as label

        Returns:
            BERT model:
            Train:
                torch.utils.data.TensorDataset
                    each record: (input_ids, input_mask, segment_ids, label)
            Test:
                torch.utils.data.TensorDataset
                    each record: (input_ids, input_mask, segment_ids)
            RNN model:
            Train:
                torch.utils.data.TensorDataset
                    each record: (s1_ids, s2_ids, s1_length, s2_length, label)
            Test:
                torch.utils.data.TensorDataset
                    each record: (s1_ids, s2_ids, s1_length, s2_length)
        """
        examples, features_list = self._load_file1(file_path, train)

        # if self.model_type == 'bert':
        dataset = self._convert_sentence_pair_to_bert_dataset(
            features_list)
        # else:  # rnn
        #     dataset = self._convert_sentence_pair_to_rnn_dataset(
        #         sc_list, bc_list, label_list)
        return examples, features_list, dataset

    def load_train_and_valid_files(self, train_file, valid_file):
        """Load all files for SMP-CAIL2020-Argmine.

        Args:
            train_file, valid_file: files for SMP-CAIL2020-Argmine

        Returns:
            train_set, valid_set_train, valid_set_valid
            all are torch.utils.data.TensorDataset
        """
        print('Loading train records for train...')
        train_exam, train_feat, train_set = self.load_file(train_file, True)
        print(len(train_set), 'training records loaded.')
        # print('Loading train records for valid...')
        # train_exam, train_feat, valid_set_train = self.load_file(train_file, False)
        # print(len(valid_set_train), 'train records loaded.')
        print('Loading valid records...')
        valid_exam, valid_feat, valid_set_valid = self.load_file(valid_file, False)
        print(len(valid_set_valid), 'valid records loaded.')
        return train_set, train_set, valid_set_valid, train_exam, valid_exam, train_feat, valid_feat

    def _load_file(self, filename, train: bool = True):
        """Load SMP-CAIL2020-Argmine train/test file.

        For train file,
        The ratio between positive samples and negative samples is 1:4
        Copy positive 3 times so that positive:negative = 1:1

        Args:
            filename: SMP-CAIL2020-Argmine file
            train:
                If True, train file with last column as label
                Otherwise, test file without last column as label

        Returns:
            sc_list, bc_list, label_list with the same length
            sc_list, bc_list: List[List[str]], list of word tokens list
            label_list: List[int], list of labels
        """
        data_frame = pd.read_csv(filename)
        sc_list, bc_list, label_list = [], [], []
        for row in data_frame.itertuples(index=False):
            # candidates = row[0:2]
            answer = bool(row[-1]) if train else None
            sc_tokens = self.tokenizer.tokenize(row[0])
            bc_tokens = self.tokenizer.tokenize(row[1])
            label = 1 if answer else 0

            sc_list.append(sc_tokens)
            bc_list.append(bc_tokens)
            if train:
                label_list.append(label)
            # for i, _ in enumerate(candidates):
            #     bc_tokens = self.tokenizer.tokenize(candidates[i])
            #     if train:
            #         if i + 1 == answer:
            #             # Copy positive sample 4 times
            #             for _ in range(len(candidates) - 1):
            #                 sc_list.append(sc_tokens)
            #                 bc_list.append(bc_tokens)
            #                 label_list.append(1)
            #         else:
            #             sc_list.append(sc_tokens)
            #             bc_list.append(bc_tokens)
            #             label_list.append(0)
            #     else:  # test
            #         sc_list.append(sc_tokens)
            #         bc_list.append(bc_tokens)
        return sc_list, bc_list, label_list

    def _load_file1(self, filename, train: bool = True):

        with open(filename, 'r', encoding='utf-8') as reader:
            full_data = json.load(reader)  # 完整的原始数据

        def is_whitespace(c):
            if c.isspace() or ord(c) == 0x202F or ord(c) == 0x2000:
                return True
            return False

        cnt = 0
        examples = []
        for case in tqdm(full_data):  # 遍历每个样本
            key = case['_id']
            qas_type = ""  # case['type']
            sup_facts = set([(sp[0], sp[1]) for sp in case['supporting_facts']])  # TODO: 为啥是个集合？为了去重？
            sup_titles = set([sp[0] for sp in case['supporting_facts']])  # sup para 的title 列表
            orig_answer_text = case['answer']

            sent_id = 0
            doc_tokens = []
            sent_names = []
            sup_facts_sent_id = []
            sent_start_end_position = []
            para_start_end_position = []
            entity_start_end_position = []
            ans_start_position, ans_end_position = [], []

            JUDGE_FLAG = orig_answer_text == 'yes' or orig_answer_text == 'no' or orig_answer_text == 'unknown' or orig_answer_text == ""  # judge_flag??
            FIND_FLAG = False

            char_to_word_offset = []  # Accumulated along all sentences
            prev_is_whitespace = True

            # for debug
            titles = set()
            para_data = case['context']
            for paragraph in para_data:  # 选中的段落
                title = paragraph[0]
                sents = paragraph[1]  # 句子列表
                # ratio = (sum([len(sent) for sent in sents]) + len(case['question'])) * 1.0 / 512
                # sents = [dynamic_fit_bert_size(sent, ratio) for sent in sents]

                titles.add(title)  # 选中的title
                is_gold_para = 1 if title in sup_titles else 0  # 是否是gold para

                para_start_position = len(doc_tokens)  # 刚开始doc_tokens是空的

                for local_sent_id, sent in enumerate(sents):  # 处理段落的每个句子
                    if local_sent_id >= max_support_sents:  # 句子数量限制：一个段落最多只允许44个句子
                        break

                    # Determine the global sent id for supporting facts
                    local_sent_name = (title, local_sent_id)  # （title， 句子在段落中的位置）
                    sent_names.append(local_sent_name)  # 作为句子的名字
                    if local_sent_name in sup_facts:
                        sup_facts_sent_id.append(sent_id)  # TODO： 这个跟原始的sup标签有啥区别
                    sent_id += 1  # 这个句子的id是以整个article为范围的，为什么?
                    sent = " ".join(sent)
                    sent += " "

                    sent_start_word_id = len(doc_tokens)  # 句子开始位置的word id
                    sent_start_char_id = len(char_to_word_offset)  # 句子开始位置的char id

                    for c in sent:  # 遍历整个句子的字符，建立char到word之间的映射关系
                        if is_whitespace(c):
                            prev_is_whitespace = True
                        else:
                            if prev_is_whitespace:
                                doc_tokens.append(c)
                            else:
                                doc_tokens[-1] += c
                            prev_is_whitespace = False
                        char_to_word_offset.append(len(doc_tokens) - 1)

                    sent_end_word_id = len(doc_tokens) - 1  # 句子结尾的word位置
                    sent_start_end_position.append((sent_start_word_id, sent_end_word_id))  # 句子开始和结束的位置，以元组形式保存

                    # Answer char position
                    answer_offsets = []
                    offset = -1

                    tmp_answer = " ".join(orig_answer_text)
                    while True:

                        offset = sent.find(tmp_answer, offset + 1)
                        if offset != -1:
                            #不在supporting_fact中则不添加到answer_offsets，去找正确的那段
                            if local_sent_id not in sup_facts_sent_id:
                                break
                            answer_offsets.append(offset)  # 把所有相同答案的开始位置都找到
                        else:
                            break

                    # answer_offsets = [m.start() for m in re.finditer(orig_answer_text, sent)]
                    if not JUDGE_FLAG and not FIND_FLAG and len(answer_offsets) > 0:
                        FIND_FLAG = True  # 标志找到了答案，TODO：这个有啥用
                        for answer_offset in answer_offsets:
                            start_char_position = sent_start_char_id + answer_offset  # 答案开始的char位置
                            end_char_position = start_char_position + len(tmp_answer) - 1  # 答案结束的char位置
                            # 答案开始的token位置，每个答案都保存
                            ans_start_position.append(char_to_word_offset[start_char_position])
                            ans_end_position.append(char_to_word_offset[end_char_position])

                    # Truncate longer document
                    if len(doc_tokens) >= 460:  # 如果大于382个词则break
                        # 这个截断会让每个段落至少有一个句子被加入，即使整个样本已经超过382，这样后面匹配entity还能匹配上吗？
                        break

                # 问题改写
                # case['question'] = dynamic_fit_bert_size(case['question'], ratio)
                if len(case['question']) > 50:
                    case['question'] = case['question'][-50:]
                para_end_position = len(doc_tokens) - 1
                # 一个段落的开始和结束token位置（白空格分词）
                para_start_end_position.append(
                    (para_start_position, para_end_position, title, is_gold_para))  # 顺便加上开始和结束位置

            if len(ans_end_position) > 1:
                cnt += 1  # 如果答案结束的位置大于1，cnt+1，如果答案结束位置是0呢？
            if key < 10:
                print("qid {}".format(key))
                print("qas type {}".format(qas_type))
                print("doc tokens {}".format(doc_tokens))
                print("question {}".format(case['question']))
                print("sent num {}".format(sent_id + 1))
                print("sup face id {}".format(sup_facts_sent_id))
                print("para_start_end_position {}".format(para_start_end_position))
                print("sent_start_end_position {}".format(sent_start_end_position))
                print("entity_start_end_position {}".format(entity_start_end_position))
                print("orig_answer_text {}".format(orig_answer_text))
                print("ans_start_position {}".format(ans_start_position))
                print("ans_end_position {}".format(ans_end_position))
            # 一个paragraph是一个example
            example = Example(
                qas_id=key,
                qas_type=qas_type,
                doc_tokens=doc_tokens,
                question_text=case['question'],
                sent_num=sent_id + 1,
                sent_names=sent_names,
                sup_fact_id=sup_facts_sent_id,
                para_start_end_position=para_start_end_position,  # 一个样本是一个article, 有多个段落开始和结束的位置
                sent_start_end_position=sent_start_end_position,
                entity_start_end_position=entity_start_end_position,
                orig_answer_text=orig_answer_text,
                start_position=ans_start_position,  # 这里是word的开始和结束位置
                end_position=ans_end_position)
            examples.append(example)

        features_list = self.convert_examples_to_features(examples, self.tokenizer, 512, 50)
        return examples, features_list

    def convert_examples_to_features(self, examples, tokenizer, max_seq_length, max_query_length):
        # max_query_length = 50

        features = []
        failed = 0
        for (example_index, example) in enumerate(tqdm(examples)):  # 遍历所有的example
            if example.orig_answer_text == 'yes':
                ans_type = 1
            elif example.orig_answer_text == 'no':
                ans_type = 2
            elif example.orig_answer_text == 'unknown':
                ans_type = 3
            else:
                ans_type = 0  # 统计answer type

            query_tokens = ["[CLS]"]
            for token in example.question_text.split(' '):
                query_tokens.extend(tokenizer.tokenize(token))
            if len(query_tokens) > max_query_length - 1:
                query_tokens = query_tokens[:max_query_length - 1]
            query_tokens.append("[SEP]")

            # para_spans = []
            # entity_spans = []
            sentence_spans = []
            all_doc_tokens = []
            orig_to_tok_index = []
            orig_to_tok_back_index = []
            tok_to_orig_index = [0] * len(query_tokens)

            all_doc_tokens = ["[CLS]"]  # 这一段不是啰嗦的代码吗
            for token in example.question_text.split(' '):
                all_doc_tokens.extend(tokenizer.tokenize(token))
            if len(all_doc_tokens) > max_query_length - 1:
                all_doc_tokens = all_doc_tokens[:max_query_length - 1]
            all_doc_tokens.append("[SEP]")

            for (i, token) in enumerate(example.doc_tokens):  # 遍历context的所有token（白空格分割）
                orig_to_tok_index.append(len(all_doc_tokens))  # 空格分词的token与wp分词后的token对应
                sub_tokens = tokenizer.tokenize(token)
                for sub_token in sub_tokens:
                    tok_to_orig_index.append(i)  # wp 分词后的token对应的空格分词的token
                    all_doc_tokens.append(sub_token)
                orig_to_tok_back_index.append(len(all_doc_tokens) - 1)  # 这个看意思应该是原始token与wp分词后的最后一个subtoken对应？

            def relocate_tok_span(orig_start_position, orig_end_position, orig_text):
                # word的（在para中的）开始和结束位置
                if orig_start_position is None:  # 如果输入的是none，返回0，实际上不会存在这种情况
                    return 0, 0

                tok_start_position = orig_to_tok_index[orig_start_position]
                if orig_end_position < len(example.doc_tokens) - 1:  # 如果结束位置没有超出了边界
                    tok_end_position = orig_to_tok_index[orig_end_position + 1] - 1
                else:
                    tok_end_position = len(all_doc_tokens) - 1  # 超出边界
                # Make answer span more accurate.
                return _improve_answer_span(
                    all_doc_tokens, tok_start_position, tok_end_position, tokenizer, orig_text)

            ans_start_position, ans_end_position = [], []
            for ans_start_pos, ans_end_pos in zip(example.start_position, example.end_position):  # 遍历每一个答案开始和结束位置
                s_pos, e_pos = relocate_tok_span(ans_start_pos, ans_end_pos, example.orig_answer_text)
                ans_start_position.append(s_pos)  # 这里返回的是答案在bert输入中的位置
                ans_end_position.append(e_pos)

            # for entity_span in example.entity_start_end_position:
            #     ent_start_position, ent_end_position \
            #         = relocate_tok_span(entity_span[0], entity_span[1], entity_span[2])
            #     entity_spans.append((ent_start_position, ent_end_position, entity_span[2], entity_span[3]))
            # 这里找到了每个实体在bert输入中的开始和结束位置

            for sent_span in example.sent_start_end_position:  # 每个句子开始和结束word的id
                if sent_span[0] >= len(orig_to_tok_index) or sent_span[0] >= sent_span[1]:
                    continue  # 如果句子的开始位置大于映射表的范围，或者开始与结束位置相同（空句子），就continue
                sent_start_position = orig_to_tok_index[sent_span[0]]  # 句子在bert输入中的开始和结束位置
                sent_end_position = orig_to_tok_back_index[
                    sent_span[1]]  # 句子结束的sub word位置（这里就是orig_to_tok_back_index的用处）
                sentence_spans.append((sent_start_position, sent_end_position))  # 句子在bert输入中的开始和结束位置

            # for para_span in example.para_start_end_position:
            #     if para_span[0] >= len(orig_to_tok_index) or para_span[0] >= para_span[1]:
            #         continue
            #     para_start_position = orig_to_tok_index[para_span[0]]
            #     para_end_position = orig_to_tok_back_index[para_span[1]]
            #     para_spans.append((para_start_position, para_end_position, para_span[2], para_span[3]))  # 3是是否是sup para

            # Padding Document
            all_doc_tokens = all_doc_tokens[:max_seq_length - 1] + ["[SEP]"]
            doc_input_ids = tokenizer.convert_tokens_to_ids(all_doc_tokens)
            query_input_ids = tokenizer.convert_tokens_to_ids(query_tokens)

            doc_input_mask = [1] * len(doc_input_ids)
            doc_segment_ids = [0] * len(query_input_ids) + [1] * (len(doc_input_ids) - len(query_input_ids))

            while len(doc_input_ids) < max_seq_length:
                doc_input_ids.append(0)
                doc_input_mask.append(0)
                doc_segment_ids.append(0)

            # Padding Question
            query_input_mask = [1] * len(query_input_ids)
            query_segment_ids = [0] * len(query_input_ids)

            while len(query_input_ids) < max_query_length:
                query_input_ids.append(0)
                query_input_mask.append(0)
                query_segment_ids.append(0)

            assert len(doc_input_ids) == max_seq_length
            assert len(doc_input_mask) == max_seq_length
            assert len(doc_segment_ids) == max_seq_length
            assert len(query_input_ids) == max_query_length
            assert len(query_input_mask) == max_query_length
            assert len(query_segment_ids) == max_query_length

            sentence_spans = get_valid_spans(sentence_spans, max_seq_length)
            # para_spans = get_valid_spans(para_spans, max_seq_length)

            sup_fact_ids = example.sup_fact_id
            sent_num = len(sentence_spans)
            sup_fact_ids = [sent_id for sent_id in sup_fact_ids if sent_id < sent_num]
            if len(sup_fact_ids) != len(example.sup_fact_id):
                failed += 1
            if example.qas_id < 10:
                print("qid {}".format(example.qas_id))
                print("all_doc_tokens {}".format(all_doc_tokens))
                print("doc_input_ids {}".format(doc_input_ids))
                print("doc_input_mask {}".format(doc_input_mask))
                print("doc_segment_ids {}".format(doc_segment_ids))
                print("query_tokens {}".format(query_tokens))
                print("query_input_ids {}".format(query_input_ids))
                print("query_input_mask {}".format(query_input_mask))
                print("query_segment_ids {}".format(query_segment_ids))
                # print("para_spans {}".format(para_spans))
                print("sentence_spans {}".format(sentence_spans))
                # print("entity_spans {}".format(entity_spans))
                print("sup_fact_ids {}".format(sup_fact_ids))
                print("ans_type {}".format(ans_type))
                print("tok_to_orig_index {}".format(tok_to_orig_index))
                print("ans_start_position {}".format(ans_start_position))
                print("ans_end_position {}".format(ans_end_position))

            features.append(
                InputFeatures(qas_id=example.qas_id,
                              doc_tokens=all_doc_tokens,
                              doc_input_ids=doc_input_ids,
                              doc_input_mask=doc_input_mask,
                              doc_segment_ids=doc_segment_ids,
                              query_tokens=query_tokens,
                              query_input_ids=query_input_ids,
                              query_input_mask=query_input_mask,
                              query_segment_ids=query_segment_ids,
                              sent_spans=sentence_spans,
                              sup_fact_ids=sup_fact_ids,
                              ans_type=ans_type,
                              token_to_orig_map=tok_to_orig_index,
                              start_position=ans_start_position,
                              end_position=ans_end_position)
            )
        return features

    def _convert_sentence_pair_to_bert_dataset(
            self, features_list):
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
        IGNORE_INDEX = -100
        sent_limit = max_support_sents
        # max_query_len  = 50

        doc_input_ids, doc_input_mask, doc_segment_ids, query_mapping = [],[],[],[]
        start_mapping, all_mapping, is_support = [], [], []
        y1, y2, ids, q_type = [], [], [], []
        tok_to_orig_index = []
        for i, features in tqdm(enumerate(features_list), ncols=80):
            doc_input_ids.append(features.doc_input_ids)
            doc_input_mask.append(features.doc_input_mask)
            doc_segment_ids.append(features.doc_segment_ids)
            query_mapping_ = torch.Tensor(self.max_seq_len)

            if len(features.token_to_orig_map) <= self.max_seq_len:
                features.token_to_orig_map = features.token_to_orig_map + [0]*(self.max_seq_len-len(features.token_to_orig_map))
            features.token_to_orig_map = features.token_to_orig_map[:self.max_seq_len]
            tok_to_orig_index.append(features.token_to_orig_map)

            start_mapping_ = torch.Tensor(sent_limit, self.max_seq_len)
            all_mapping_ = torch.Tensor(self.max_seq_len, sent_limit)
            is_support_ = [0] * sent_limit


            for mapping in [start_mapping_, all_mapping_, query_mapping_]:
                mapping.zero_()  # 把几个mapping都初始化为0

            for j in range(features.sent_spans[0][0] - 1):
                query_mapping_[j] = 1

            query_mapping.append(query_mapping_.unsqueeze(dim=0))

            if features.ans_type == 0:
                if len(features.end_position) == 0:
                    y1.append(0)
                    y2.append(0)  # 如果结束位置是0，span的标签就为0
                elif features.end_position[0] < self.max_seq_len:
                    y1.append(features.start_position[0])  # 只用第一个找到的span
                    y2.append(features.end_position[0])
                else:
                    y1.append(0)
                    y2.append(0)
            else:
                y1.append(IGNORE_INDEX)  # span是-100
                y2.append(IGNORE_INDEX)
            q_type.append(features.ans_type)  # 这个明明是answer_type，非要叫q_type
            ids.append(features.qas_id)

            for j, sent_span in enumerate(features.sent_spans[:sent_limit]):  # 句子序号，span
                is_sp_flag = j in features.sup_fact_ids  # 这个代码写的真几把烂#我也觉得
                start, end = sent_span
                # if start < end:  # 还有start大于end的时候？
                is_support_[j] = int(is_sp_flag)  # 样本i的第j个句子是否是sp
                all_mapping_[start:end + 1, j] = 1 # （batch_size, max_seq_len, 20) 第j个句子开始和结束全为1
                start_mapping_[j, start] = 1    # （batch_size, 20, max_seq_len)
            is_support.append(is_support_)
            start_mapping.append(start_mapping_.unsqueeze(dim=0))
            all_mapping.append(all_mapping_.unsqueeze(dim=0))


        context_idxs = torch.tensor(doc_input_ids, dtype=torch.long)
        context_mask = torch.tensor(doc_input_mask, dtype=torch.long)
        segment_idxs = torch.tensor(doc_segment_ids, dtype=torch.long)
        tok_to_orig_index = torch.tensor(tok_to_orig_index, dtype=torch.long)

        query_mapping = torch.cat(query_mapping, dim=0)
        start_mapping = torch.cat(start_mapping, dim=0)
        all_mapping = torch.cat(all_mapping, dim=0)

        ids = torch.tensor(ids, dtype=torch.long)
        y1 = torch.tensor(y1, dtype=torch.long)
        y2 = torch.tensor(y2, dtype=torch.long)
        q_type = torch.tensor(q_type, dtype=torch.long)
        is_support = torch.tensor(is_support, dtype=torch.long)


        return TensorDataset(
            context_idxs, context_mask, segment_idxs,
            query_mapping, all_mapping,
            ids, y1, y2, q_type,
            start_mapping,
            is_support,tok_to_orig_index
        )

    def _convert_sentence_pair_to_rnn_dataset(
            self, s1_list, s2_list, label_list=None):
        """Convert sentences pairs to dataset for RNN model.

        Args:
            sc_list, bc_list: List[List[str]], list of word tokens list
            label_list: train: List[int], list of labels
                        test: []

        Returns:
            Train:
            torch.utils.data.TensorDataset
                each record: (s1_ids, s2_ids, s1_length, s2_length, label)
            Test:
            torch.utils.data.TensorDataset
                each record: (s1_ids, s2_ids, s1_length, s2_length, label)
        """
        all_s1_ids, all_s2_ids = [], []
        all_s1_lengths, all_s2_lengths = [], []
        for i in tqdm(range(len(s1_list)), ncols=80):
            tokens_s1, tokens_s2 = s1_list[i], s2_list[i]
            all_s1_lengths.append(min(len(tokens_s1), self.max_seq_len))
            all_s2_lengths.append(min(len(tokens_s2), self.max_seq_len))
            if len(tokens_s1) > self.max_seq_len:
                tokens_s1 = tokens_s1[:self.max_seq_len]
            if len(tokens_s2) > self.max_seq_len:
                tokens_s2 = tokens_s2[:self.max_seq_len]
            s1_ids = self.tokenizer.convert_tokens_to_ids(tokens_s1)
            s2_ids = self.tokenizer.convert_tokens_to_ids(tokens_s2)
            if len(s1_ids) < self.max_seq_len:
                s1_ids += [0] * (self.max_seq_len - len(s1_ids))
            if len(s2_ids) < self.max_seq_len:
                s2_ids += [0] * (self.max_seq_len - len(s2_ids))
            all_s1_ids.append(s1_ids)
            all_s2_ids.append(s2_ids)
        all_s1_ids = torch.tensor(all_s1_ids, dtype=torch.long)
        all_s2_ids = torch.tensor(all_s2_ids, dtype=torch.long)
        all_s1_lengths = torch.tensor(all_s1_lengths, dtype=torch.long)
        all_s2_lengths = torch.tensor(all_s2_lengths, dtype=torch.long)
        if label_list:  # train
            all_label_ids = torch.tensor(label_list, dtype=torch.long)
            return TensorDataset(
                all_s1_ids, all_s2_ids, all_s1_lengths, all_s2_lengths,
                all_label_ids)
        # test
        return TensorDataset(
            all_s1_ids, all_s2_ids, all_s1_lengths, all_s2_lengths)


def test_data():
    """Test for data module."""
    # For BERT model
    data = Data('model/bert/vocab.txt', model_type='bert')
    _, _, _ = data.load_train_and_valid_files(
        'SMP-CAIL2020-train.csv',
        'SMP-CAIL2020-test1.csv')
    # For RNN model
    data = Data('model/rnn/vocab.txt', model_type='rnn')
    _, _, _ = data.load_train_and_valid_files(
        'SMP-CAIL2020-train.csv',
        'SMP-CAIL2020-test1.csv')


if __name__ == '__main__':
    test_data()

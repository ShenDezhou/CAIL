"""Data processor for SMP-CAIL2020-Argmine.

Author: Yixu GAO (yxgao19@fudan.edu.cn)

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
from typing import List
import lawa
import torch

import pandas as pd

from torch.utils.data import TensorDataset
from transformers import BertTokenizer, AutoTokenizer
# from pytorch_pretrained_bert import BertTokenizer
from tqdm import tqdm


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
        return lawa.lcut(sentence)

    def convert_tokens_to_ids(
            self, tokens_list: List[str]) -> List[int]:
        """Convert tokens to ids.

        Args:
            tokens_list: word list

        Returns:
            index list
        """
        return [self.dictionary.get(w, 1) for w in tokens_list]


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
        if 'bert' == self.model_type:
            self.tokenizer = BertTokenizer.from_pretrained(config.bert_model_path)#BertTokenizer(vocab_file)
        elif 'bertxl' == self.model_type:
            self.tokenizer = AutoTokenizer.from_pretrained(config.bert_model_path)
        else:  # rnn
            self.tokenizer = Tokenizer(vocab_file)
        self.max_seq_len = max_seq_len
        self.config = config

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
        sc_list, label_list = self._load_file(file_path, train)
        if 'bert' == self.model_type:
            dataset = self._convert_sentence_pair_to_bert_dataset(
                sc_list,  label_list)
        elif 'bertxl' == self.model_type:
            dataset = self._convert_sentence_pair_to_bertxl_dataset(
                sc_list,  label_list)
        else:  # rnn
            dataset = self._convert_sentence_pair_to_rnn_dataset(
                sc_list,  label_list)
        return dataset

    def load_train_and_valid_files(self, train_file, valid_file, isTPU=False):
        """Load all files for SMP-CAIL2020-Argmine.

        Args:
            train_file, valid_file: files for SMP-CAIL2020-Argmine

        Returns:
            train_set, valid_set_train, valid_set_valid
            all are torch.utils.data.TensorDataset
        """
        print('Loading train records for train...')
        train_set = self.load_file(train_file, True)
        print(len(train_set), 'training records loaded.')
        if 'bert' in self.model_type:
            valid_set_train = None
        else:
            print('Loading train records for valid...')
            valid_set_train = self.load_file(train_file, False)
            print(len(valid_set_train), 'train records loaded.')
        print('Loading valid records...')
        if isTPU:
            valid_set_valid = self.load_file(valid_file, True)
        else:
            valid_set_valid = self.load_file(valid_file, False)
        print(len(valid_set_valid), 'valid records loaded.')
        return train_set, valid_set_train, valid_set_valid

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

        sc_list, label_list = [], []
        for row in data_frame.itertuples(index=False):
            # top 800 characters are processed.
            sc_tokens = self.tokenizer.tokenize(str(row[1]))
            # bc_tokens = self.tokenizer.tokenize(str(row[2]))
            if train:
                sc_list.append(sc_tokens)
                # bc_list.append(bc_tokens)
                label_list.append(row[0])
            else:  # test
                sc_list.append(sc_tokens)
                # bc_list.append(bc_tokens)
                # label_list.append(row[0])

        return sc_list, label_list

    def _convert_sentence_pair_to_bert_dataset(
            self, s1_list, label_list=None):
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
        for i, _ in tqdm(enumerate(s1_list), ncols=80):
            tokens = ['[CLS]'] + s1_list[i] + ['[SEP]']
            segment_ids = [0] * len(tokens)
            # tokens += s2_list[i] + ['[SEP]']
            # segment_ids += [1] * (len(s2_list[i]) + 1)
            if len(tokens) > self.max_seq_len :
                half = self.max_seq_len // 2
                tokens = tokens[:half] + tokens[-half:]
                assert len(tokens) ==  self.max_seq_len
                segment_ids = segment_ids[:half] + segment_ids[-half:]
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)
            tokens_len = len(input_ids)
            input_ids += [0] * ( self.max_seq_len - tokens_len)
            segment_ids += [0] * ( self.max_seq_len - tokens_len)
            input_mask += [0] * ( self.max_seq_len - tokens_len)
            all_input_ids.append(input_ids)
            all_input_mask.append(input_mask)
            all_segment_ids.append(segment_ids)
        all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
        all_input_mask = torch.tensor(all_input_mask, dtype=torch.long)
        all_segment_ids = torch.tensor(all_segment_ids, dtype=torch.long)
        if label_list:  # train
            all_label_ids = torch.tensor(label_list, dtype=torch.long)
            return TensorDataset(
                all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        # test
        return TensorDataset(
            all_input_ids, all_input_mask, all_segment_ids)

    def _convert_sentence_pair_to_bertxl_dataset(
            self, s1_list,  label_list=None):
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
        all_label_list = []
        for i, _ in tqdm(enumerate(s1_list), ncols=80):
            tokens = s1_list[i]
            segment_ids = [0] * len(tokens)
            # tokens += s2_list[i] + ['[SEP]']
            # segment_ids += [1] * (len(s2_list[i]) + 1)

            if len(tokens) > self.max_seq_len:
                tokens = tokens[:self.max_seq_len//2] + tokens[-self.max_seq_len//2:]
                assert len(tokens) == self.max_seq_len
                segment_ids = segment_ids[:self.max_seq_len//2] + segment_ids[-self.max_seq_len//2:]



            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)
            tokens_len = len(input_ids)
            input_ids += [0] * (self.max_seq_len - tokens_len)
            segment_ids += [0] * (self.max_seq_len - tokens_len)
            input_mask += [0] * (self.max_seq_len - tokens_len)


            all_input_ids.append(input_ids)
            all_input_mask.append(input_mask)
            all_segment_ids.append(segment_ids)
            #
            # if label_list:  # train
            #     label_list_ = label_list[i]
            #     # if len(label_list_) < self.max_seq_len:
            #     #     label_list_ += [0] * (self.max_seq_len - tokens_len)
            #     # elif len(label_list_) > self.max_seq_len:
            #     #     label_list_ = label_list_[:self.max_seq_len // 2] + label_list_[-self.max_seq_len // 2:]
            #
            #     all_label_list.append(label_list_)


        all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
        all_input_mask = torch.tensor(all_input_mask, dtype=torch.long)
        all_segment_ids = torch.tensor(all_segment_ids, dtype=torch.long)

        if label_list:  # train
            all_label_ids = torch.tensor(label_list, dtype=torch.float)
            return TensorDataset(
                all_input_ids, all_input_mask, all_segment_ids,
                all_label_ids)
        # test
        return TensorDataset(
            all_input_ids, all_input_mask, all_segment_ids)

    def _convert_sentence_pair_to_rnn_dataset(
            self, s1_list,  label_list=None):
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
        all_s1_ids= []
        all_s1_lengths= []
        for i, _ in tqdm(enumerate(s1_list), ncols=80):
            tokens_s1 = s1_list[i]
            if len(tokens_s1) > self.max_seq_len:
                tokens_s1 = tokens_s1[:self.max_seq_len]
            all_s1_lengths.append(len(tokens_s1))
            # if len(tokens_s2) > self.max_seq_len:
            #     tokens_s2 = tokens_s2[:self.max_seq_len]
            s1_ids = self.tokenizer.convert_tokens_to_ids(tokens_s1)
            # s2_ids = self.tokenizer.convert_tokens_to_ids(tokens_s2)
            if len(s1_ids) < self.max_seq_len:
                s1_ids += [0] * (self.max_seq_len - len(s1_ids))
            # if len(s2_ids) < self.max_seq_len:
            #     s2_ids += [0] * (self.max_seq_len - len(s2_ids))
            all_s1_ids.append(s1_ids)
            # all_s2_ids.append(s2_ids)
        print(all_s1_ids[-1])
        all_s1_ids = torch.tensor(all_s1_ids, dtype=torch.long)
        # all_s2_ids = torch.tensor(all_s2_ids, dtype=torch.long)
        all_s1_lengths = torch.tensor(all_s1_lengths, dtype=torch.long)
        # all_s2_lengths = torch.tensor(all_s2_lengths, dtype=torch.long)
        if label_list:  # train
            all_label_ids = torch.tensor(label_list, dtype=torch.long)
            print(all_s1_lengths)
            return TensorDataset(
                all_s1_ids, all_s1_lengths,
                all_label_ids)
        # test
        return TensorDataset(all_s1_ids, all_s1_lengths)


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

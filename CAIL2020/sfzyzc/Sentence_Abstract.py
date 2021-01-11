"""Test model for SMP-CAIL2020-Argmine.

Author: Tsinghuaboy tsinghua9boy@sina.com

Usage:
    python main.py --model_config 'config/bert_config.json' \
                   --in_file 'data/SMP-CAIL2020-test1.csv' \
                   --out_file 'bert-submission-test-1.csv'
    python main.py --model_config 'config/rnn_config.json' \
                   --in_file 'data/SMP-CAIL2020-test1.csv' \
                   --out_file 'rnn-submission-test-1.csv'
"""
import argparse
import json
import os
import time
from types import SimpleNamespace

import fire
import pandas
import pandas as pd
import torch
from torch.utils.data import DataLoader

from sfzyzb.data import Data
from sfzyzb.evaluate import evaluate
from sfzyzb.model import BertForClassification, RnnForSentencePairClassification, LogisticRegression
from sfzyzb.utils import load_torch_model



LABELS = ['0', '1']
MODEL_MAP = {
    'bert': BertForClassification,
    'rnn': RnnForSentencePairClassification,
    'lr': LogisticRegression
}

class Sentence_Abstract(object):

    def __init__(self,model_config='sfzyzb/config/bert_config-l.json'):
        # 0. Load config
        with open(model_config) as fin:
            config = json.load(fin, object_hook=lambda d: SimpleNamespace(**d))
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
            # device = torch.device('cpu')
        else:
            self.device = torch.device('cpu')

        # 1. Load data
        self.data = Data(vocab_file=os.path.join(config.model_path, 'vocab.txt'),
                    max_seq_len=config.max_seq_len,
                    model_type=config.model_type, config=config)

        # 2. Load model
        self.model = MODEL_MAP[config.model_type](config)
        self.model = load_torch_model(
            self.model, model_path=os.path.join(config.model_path, 'model.bin'))
        self.model.to(self.device)
        self.config = config


    def get_abstract(self, in_file):
        # 0. preprocess file
        tag_sents = []
        para_id = 0

        convert_file ="data/{}-2.csv".format(time.time())
        with open(in_file, 'r', encoding='utf-8') as fin:
            for line in fin:
                sents = json.loads(line.strip())
                text = sents['text']
                sentences = [item['sentence'] for item in text]
                for sent in sentences:
                    tag_sents.append((para_id, sent))
                para_id += 1
            df = pandas.DataFrame(tag_sents, columns=['para', 'content'])
            df.to_csv(convert_file, columns=['para', 'content'], index=False)

        test_set = self.data.load_file(convert_file, train=False)
        data_loader_test = DataLoader(
            test_set, batch_size=self.config.batch_size, shuffle=False)

        # 3. Evaluate
        answer_list = evaluate(self.model, data_loader_test, self.device)
        # 4. Write answers to file
        # df = pd.read_csv("data/para_content_test.csv")
        idcontent_list = list(df.itertuples(index=False))
        filter_list = [k for k, v in zip(idcontent_list, answer_list) if v]
        df = pd.DataFrame(filter_list, columns=['para', 'content'])
        out_file = "data/{}-3.csv".format(time.time())
        df.to_csv(out_file, columns=['para', 'content'], index=False)
        return out_file



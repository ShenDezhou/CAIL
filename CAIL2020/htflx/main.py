"""Test model for SMP-CAIL2020-Argmine.

Author: Yixu GAO yxgao19@fudan.edu.cn

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
from types import SimpleNamespace

import fire
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data import Data
from evaluate import evaluate
from model import BertYForClassification, RnnForSentencePairClassification, LogisticRegression
from utils import load_torch_model



LABELS = ['1', '2', '3', '4', '5']
MODEL_MAP = {
    'bert': BertYForClassification,
    'rnn': RnnForSentencePairClassification,
    'lr': LogisticRegression
}


def main(in_file='/data/SMP-CAIL2020-test1.csv',
         out_file='/output/result1.csv',
         model_config='config/bert_config.json'):
    """Test model for given test set on 1 GPU or CPU.

    Args:
        in_file: file to be tested
        out_file: output file
        model_config: config file
    """
    # 0. Load config
    with open(model_config) as fin:
        config = json.load(fin, object_hook=lambda d: SimpleNamespace(**d))
    if torch.cuda.is_available():
        #device = torch.device('cuda')
        device = torch.device('cpu')
    else:
        device = torch.device('cpu')
    # 1. Load data
    data = Data(vocab_file=os.path.join(config.model_path, 'vocab.txt'),
                max_seq_len=config.max_seq_len,
                model_type=config.model_type, config=config)
    test_set = data.load_file(in_file, train=False)
    data_loader_test = DataLoader(
        test_set, batch_size=config.batch_size, shuffle=False)
    # 2. Load model
    model = MODEL_MAP[config.model_type](config)
    model = load_torch_model(
        model, model_path=os.path.join(config.model_path, 'model.bin'))
    model.to(device)
    # 3. Evaluate
    answer_list = evaluate(model, data_loader_test, device)
    # 4. Write answers to file
    id_list = pd.read_csv(in_file)['id'].tolist()
    with open(out_file, 'w') as fout:
        fout.write('id,answer\n')
        for i, j in zip(id_list, answer_list):
            fout.write(str(i) + ',' + str(j) + '\n')


if __name__ == '__main__':
    fire.Fire(main)

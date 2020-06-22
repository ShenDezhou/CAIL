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

import pandas as pd
import torch
from torch.utils.data import DataLoader

from preprocess import preprocess
from data import Data
from evaluate import evaluatex
from model import BertXForClassification, RnnForSentencePairClassification
from utils import load_torch_model


LABELS = ['0', '1']
MODEL_MAP = {
    'bert': BertXForClassification,
    'rnn': RnnForSentencePairClassification
}

TEMPFILE='test.csv'

def main(in_file='/input/',
         out_file='/output/result.csv',
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
        device = torch.device('cuda')
        # device = torch.device('cpu')
    else:
        device = torch.device('cpu')
    # 1. Load data
    data = Data(vocab_file=os.path.join(config.model_path, 'vocab.txt'),
                max_seq_len=config.max_seq_len,
                model_type=config.model_type, config=config)

    # 1.1 preprocess '/input/' to 'test.csv' file.
    preprocess(in_file, TEMPFILE)
    test_set = data.load_file(TEMPFILE, train=False)
    data_loader_test = DataLoader(
        test_set, batch_size=config.batch_size, shuffle=False)
    # 2. Load model
    model = MODEL_MAP[config.model_type](config)
    model = load_torch_model(
        model, model_path=os.path.join(config.model_path, 'model.bin'))
    model.to(device)
    # 3. Evaluate
    answer_list = evaluatex(model, data_loader_test, device)
    # 4. Write answers to file
    id_list = pd.read_csv(TEMPFILE)['id'].tolist()
    result = {}
    for i, j in zip(id_list, answer_list):
        if i not in result.keys():
            counter = 0
            result[i] = []
        if j == '1':
            result[i].append(chr(ord('A')+counter))
        counter+=1
    json.dump(result, open(out_file, "w", encoding="utf8"), indent=2, ensure_ascii=False, sort_keys=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', '-c', default='config/bert_config.json', help="specific config file", required=False)
    parser.add_argument('--in_file', '-i',  default='/input', help="input folder", required=False)
    parser.add_argument('--out_file', '-o', default='/output/result.csv', help="result file path", required=False)
    args = parser.parse_args()
    main(args.in_file, args.out_file, args.model_config)


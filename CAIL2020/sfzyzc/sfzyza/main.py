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
import re
from types import SimpleNamespace

import fire
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data import Data
from evaluate import evaluate
from model import BertForClassification,BertXLForClassification, RnnForSentencePairClassification, LogisticRegression
from utils import load_torch_model



LABELS = ['1', '2', '3', '4', '5']
MODEL_MAP = {
    'bert': BertForClassification,
    'bertxl': BertXLForClassification,
    'rnn': RnnForSentencePairClassification,
    'lr': LogisticRegression
}

def remove(text):
    cleanr = re.compile(r"[ !#\$%&'\(\)*\+,-./:;<=>?@\^_`{|}~“”？！【】（）、’‘…￥·]*")
    cleantext = re.sub(cleanr, '', text)
    return cleantext

def main(in_file='/data/SMP-CAIL2020-test1.csv',
         temp_file="data/para_content_test.csv",
         out_file='/output/result1.csv',
         model_config='config/bert_config-xl.json'):
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

    #0. preprocess file
    id_list = []
    with open(in_file, 'r', encoding='utf-8') as fin:
        for line in fin:
            sents = json.loads(line.strip())
            id = sents['id']
            id_list.append(id)
    id_dict = dict(zip(range(len(id_list)), id_list))

    # 1. Load data
    data = Data(vocab_file=os.path.join(config.model_path, 'vocab.txt'),
                max_seq_len=config.max_seq_len,
                model_type=config.model_type, config=config)
    test_set = data.load_file(temp_file, train=False)
    data_loader_test = DataLoader(
        test_set, batch_size=config.batch_size, shuffle=False)
    # 2. Load model
    model = MODEL_MAP[config.model_type](config)
    model = load_torch_model(
        model, model_path=os.path.join(config.model_path, 'model.bin'))
    model.to(device)
    # 3. Evaluate
    answer_list = evaluate(model, data_loader_test, device)
    token_list = []
    for line in answer_list:
        tokens = data.tokenizer.decode(line, skip_special_tokens=True)
        token_list.append(tokens)
    # 4. Write answers to file
    para_list = pd.read_csv(temp_file)['para'].to_list()
    summary_dict = dict(zip(id_dict.values(), [""] * len(id_dict)))

    result = zip(para_list, token_list)
    for id, summary in result:
        summary_dict[id_dict[id]] += remove(summary).replace(" ","")

    with open(out_file, 'w', encoding='utf8') as fout:
        for id, summary in summary_dict.items():
            fout.write(json.dumps({'id':id,'summary':summary},  ensure_ascii=False) + '\n')


if __name__ == '__main__':
    fire.Fire(main)

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
import itertools
import json
import os
import re
from types import SimpleNamespace

import fire
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data import Data
from evaluate import evaluate, handy_tool, calculate_accuracy_f1
from model import BERNet, BERXLNet, NERNet, NERWNet
from utils import load_torch_model



LABELS = ['1', '2', '3', '4', '5']

MODEL_MAP = {
    'bert': BERNet,
    'bertxl': BERXLNet,
    'rnn': NERNet,
    'rnnkv': NERWNet
}


def result_to_json(string, tags):
    item = {"string": string, "entities": []}
    entity_name = ""
    entity_start = 0
    idx = 0
    i = -1
    zipped = zip(string, tags)
    listzip = list(zipped)
    last = len(listzip)
    for char, tag in listzip:
        i += 1
        if tag == 3:
            item["entities"].append({"word": char, "start": idx, "end": idx+1, "type":'s'})
        elif tag == 0:
            entity_name += char
            entity_start = idx
        elif tag == 1:
            if (entity_name != "") and (i == last):
                entity_name += char
                item["entities"].append({"word": entity_name, "start": entity_start, "end": idx + 1, "type": 'bms'})
                entity_name = ""
            else:
                entity_name += char
        elif tag == 2:  # or i == len(zipped)
            entity_name += char
            item["entities"].append({"word": entity_name, "start": entity_start, "end": idx + 1, "type": 'bms'})
            entity_name = ""
        else:
            entity_name = ""
            entity_start = idx
        idx += 1
    return item

def remove(text):
    cleanr = re.compile(r"[ !#\$%&'\(\)*\+,-./:;<=>?@\^_`{|}~“”？！【】（）、’‘…￥·]*")
    cleantext = re.sub(cleanr, '', text)
    return cleantext

def main(out_file='output/result.txt',
         model_config='config/rnn_config.json'):
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
    # id_list = []
    # with open(in_file, 'r', encoding='utf-8') as fin:
    #     for line in fin:
    #         sents = json.loads(line.strip())
    #         id = sents['id']
    #         id_list.append(id)
    # id_dict = dict(zip(range(len(id_list)), id_list))

    # 1. Load data
    data = Data(vocab_file=os.path.join(config.model_path, 'vocab.txt'),
                max_seq_len=config.max_seq_len,
                model_type=config.model_type, config=config)
    test_set, sc_list, label_list = data.load_file(config.test_file_path, train=True)

    token_list = []
    for line in sc_list:
        tokens = data.tokenizer.convert_ids_to_tokens(line)
        token_list.append(tokens)

    data_loader_test = DataLoader(
        test_set, batch_size=config.batch_size, shuffle=False)
    # 2. Load model
    model = MODEL_MAP[config.model_type](config)
    model = load_torch_model(
        model, model_path=os.path.join(config.model_path, 'model.bin'))
    model.to(device)
    # 3. Evaluate
    answer_list, length_list = evaluate(model, data_loader_test, device, isTest=False)

    def flatten(ll):
        return list(itertools.chain(*ll))

    train_answers = handy_tool(label_list, length_list) #gold
    #answer_list = handy_tool(answer_list, length_list) #prediction
    train_answers = flatten(train_answers)
    train_predictions = flatten(answer_list)

    train_acc, train_f1 = calculate_accuracy_f1(
        train_answers, train_predictions)
    print(train_acc, train_f1)
    mod_tokens_list = handy_tool(token_list, length_list)
    result = [result_to_json(t, s) for t,s in zip(mod_tokens_list, answer_list)]

    # 4. Write answers to file
    with open(out_file, 'w', encoding='utf8') as fout:
        for item in result:
            entities = item['entities']
            words = [d['word'] for d in entities]
            fout.write(" ".join(words) + "\n")

    # para_list = pd.read_csv(temp_file)['para'].to_list()
    # summary_dict = dict(zip(id_dict.values(), [""] * len(id_dict)))
    #
    # result = zip(para_list, token_list)
    # for id, summary in result:
    #     summary_dict[id_dict[id]] += remove(summary).replace(" ","")
    #
    # with open(out_file, 'w', encoding='utf8') as fout:
    #     for id, sumamry in summary_dict.items():
    #         fout.write(json.dumps({'id':id,'summary':sumamry},  ensure_ascii=False) + '\n')


if __name__ == '__main__':
    fire.Fire(main)

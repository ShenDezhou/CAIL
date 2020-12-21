
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
from model import BertForClassification, RnnForSentencePairClassification, FullyConnectNet, LogisticRegression, CharCNN
from utils import load_torch_model



MODEL_MAP = {
    'bert': BertForClassification,
    'bertxl': BertForClassification,
    'rnn': RnnForSentencePairClassification,
    'lr': LogisticRegression,
    'sg': FullyConnectNet,
    'cnn': CharCNN
}

LABELS = ['1', '2']


def get_prf(res):
    # According to https://github.com/dice-group/gerbil/wiki/Precision,-Recall-and-F1-measure
    if res["TP"] == 0:
        if res["FP"] == 0 and res["FN"] == 0:
            precision = 1.0
            recall = 1.0
            f1 = 1.0
        else:
            precision = 0.0
            recall = 0.0
            f1 = 0.0
    else:
        precision = 1.0 * res["TP"] / (res["TP"] + res["FP"])
        recall = 1.0 * res["TP"] / (res["TP"] + res["FN"])
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def gen_micro_macro_result(res):
    precision = []
    recall = []
    f1 = []
    total = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    for a in range(0, len(res)):
        total["TP"] += res[a]["TP"]
        total["FP"] += res[a]["FP"]
        total["FN"] += res[a]["FN"]
        total["TN"] += res[a]["TN"]

        p, r, f = get_prf(res[a])
        precision.append(p)
        recall.append(r)
        f1.append(f)

    micro_precision, micro_recall, micro_f1 = get_prf(total)

    macro_precision = 0
    macro_recall = 0
    macro_f1 = 0
    for a in range(0, len(f1)):
        macro_precision += precision[a]
        macro_recall += recall[a]
        macro_f1 += f1[a]

    macro_precision /= len(f1)
    macro_recall /= len(f1)
    macro_f1 /= len(f1)

    return {
        "micro_precision": round(micro_precision, 3),
        "micro_recall": round(micro_recall, 3),
        "micro_f1": round(micro_f1, 3),
        "macro_precision": round(macro_precision, 3),
        "macro_recall": round(macro_recall, 3),
        "macro_f1": round(macro_f1, 3)
    }


def single_label_accuracy(prediction, label, num_class, result):
    while len(result) <= num_class:
        result.append({"TP": 0, "FN": 0, "FP": 0, "TN": 0})
    for a in range(0, len(prediction)):
        it_is = int(prediction[a])
        should_be = int(label[a])
        if it_is == should_be:
            result[it_is]["TP"] += 1
        else:
            result[it_is]["FP"] += 1
            result[should_be]["FN"] += 1
    return result


def main(in_file='data/f_test.csv',
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
        device = torch.device('cuda')
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
    id_list = pd.read_csv(in_file)['accusation'].tolist()
    result = []
    result = single_label_accuracy(answer_list, id_list, config.num_classes, result)
    metrics = gen_micro_macro_result(result)
    print(metrics)


if __name__ == '__main__':
    fire.Fire(main)

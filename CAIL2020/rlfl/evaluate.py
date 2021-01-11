"""Evaluate model and calculate results for SMP-CAIL2020-Argmine.

Author: Tsinghuaboy tsinghua9boy@sina.com
"""
from typing import List
import codecs

import pandas
import torch

from tqdm import tqdm
from sklearn import metrics
from classmerge import classy_dic, indic

LABELS = ['1','2']


def calculate_accuracy_f1(
        golds: List[str], predicts: List[str]) -> tuple:
    """Calculate accuracy and f1 score.

    Args:
        golds: answers
        predicts: predictions given by model

    Returns:
        accuracy, f1 score
    """
    return metrics.accuracy_score(golds, predicts), \
           metrics.f1_score(
               golds, predicts,
               labels=LABELS, average='macro')


# def get_labels_from_file(filename):
#     """Get labels on the last column from file.
#
#     Args:
#         filename: file name
#
#     Returns:
#         List[str]: label list
#     """
#     labels = []
#     with codecs.open(filename, 'r', encoding='utf-8') as fin:
#         fin.readline()
#         for line in fin:
#             labels.append(line.strip().split(',')[-1])
#     return labels

def get_labels_from_file(filename):
    """Get labels on the last column from file.

    Args:
        filename: file name

    Returns:
        List[str]: label list
    """
    data_frame = pandas.read_csv(filename)
    labels = data_frame['type1'].map(indic).tolist()
    return labels

def eval_file(golds_file, predicts_file):
    """Evaluate submission file

    Args:
        golds_file: file path
        predicts_file:  file path

    Returns:
        accuracy, f1 score
    """
    golds = get_labels_from_file(golds_file)
    predicts = get_labels_from_file(predicts_file)
    return calculate_accuracy_f1(golds, predicts)


def evaluate(model, data_loader, device) -> List[str]:
    """Evaluate model on data loader in device.

    Args:
        model: model to be evaluate
        data_loader: torch.utils.data.DataLoader
        device: cuda or cpu

    Returns:
        answer list
    """
    model.eval()
    outputs = torch.tensor([], dtype=torch.float).to(device)
    for batch in tqdm(data_loader, desc='Evaluation', ncols=80):
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            logits = model(*batch)
        outputs = torch.cat([outputs, logits[:, :]])
    answer_list = []
    for i in range(len(outputs)):
        logits = outputs[i]
        answer = int(torch.argmax(logits, dim=-1))
        answer_list.append(answer)
    return answer_list


if __name__ == '__main__':
    acc, f1_score = eval_file(
        'data/train.csv', 'rule_baseline/submission.csv')
    print("acc: {}, f1: {}".format(acc, f1_score))

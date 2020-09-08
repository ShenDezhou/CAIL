import argparse
import json
import os
from types import SimpleNamespace

import torch

from transformers import (
    CONFIG_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    LineByLineTextDataset,
    PreTrainedTokenizer,
    TextDataset,
    Trainer,
    TrainingArguments,
    set_seed,
    BertTokenizer
)


INPUT='data/train.txt'
TEMP="temp/"
OUTPUT="modelcpu/"

parser = argparse.ArgumentParser()
parser.add_argument(
    '-c', '--config_file', default='config/bert_verify_config.json',
    help='model config file')

# parser.add_argument(
#     '--local_rank', default=0,
#     help='used for distributed parallel')
args = parser.parse_args()
with open(args.config_file) as fin:
    config = json.load(fin, object_hook=lambda d: SimpleNamespace(**d))

bert_config = AutoConfig.from_pretrained(config.bert_model_path, cache_dir=TEMP)
# main(args.config_file)
WRAPPED_MODEL = AutoModelWithLMHead.from_pretrained(
            config.bert_model_path,
            from_tf=False,
            config=bert_config,
            cache_dir=TEMP,
        )

tokenizer = BertTokenizer.from_pretrained(config.bert_model_path)

print('DONE')
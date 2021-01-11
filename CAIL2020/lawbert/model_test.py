import argparse
import json
import os
from types import SimpleNamespace

import torch
from transformers import DataCollatorForLanguageModeling
from transformers import LineByLineTextDataset
from transformers import Trainer, TrainingArguments

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.utils.utils as xu

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
WRAPPED_MODEL.to('cpu')
xm.save(WRAPPED_MODEL.state_dict(), os.path.join(OUTPUT, 'pytorch_model.bin'))
# WRAPPED_MODEL.save_pretrained(OUTPUT)
print('DONE')
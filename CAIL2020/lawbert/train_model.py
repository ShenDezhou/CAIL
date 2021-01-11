import argparse
import json
import os
from types import SimpleNamespace

import torch
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
    BertConfig,
    RobertaConfig,
    AutoModelWithLMHead,
    BertForMaskedLM,
    RobertaForMaskedLM,
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


TEMP="temp/"

parser = argparse.ArgumentParser()
parser.add_argument(
    '-c', '--config_file', default='config/bert_config.json',
    help='model config file')

args = parser.parse_args()
with open(args.config_file) as fin:
    config = json.load(fin, object_hook=lambda d: SimpleNamespace(**d))

bert_config = BertConfig.from_pretrained(config.bert_model_path, cache_dir=TEMP)

WRAPPED_MODEL = BertForMaskedLM.from_pretrained(
            config.bert_model_path,
            from_tf=False,
            config=bert_config,
            cache_dir=TEMP,
        )
for param in WRAPPED_MODEL.parameters():
    param.requires_grad = True
WRAPPED_MODEL.train()

tokenizer = BertTokenizer.from_pretrained(config.bert_model_path)
WRAPPED_MODEL.resize_token_embeddings(len(tokenizer))

print("dataset maxl:", config.max_seq_len)

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=config.train_file_path,
    block_size=config.max_seq_len,
)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

"""### Finally, we are all set to initialize our Trainer"""



training_args = TrainingArguments(
    output_dir=TEMP,
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=config.batch_size,
    save_steps=10_000,
    save_total_limit=2,
    tpu_num_cores=8,
)

trainer = Trainer(
    model=WRAPPED_MODEL,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
    prediction_loss_only=True,
)


trainer.train(model_path=config.bert_model_path)
WRAPPED_MODEL.to('cpu')
trainer.save_model(output_dir=config.trained_model_path)
torch.save(WRAPPED_MODEL.state_dict(), os.path.join(config.trained_model_path, 'pytorch_model.bin'))


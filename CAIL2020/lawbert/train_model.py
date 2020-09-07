import argparse
import json
from types import SimpleNamespace

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


TEMP="temp/"

parser = argparse.ArgumentParser()
parser.add_argument(
    '-c', '--config_file', default='config/bert_config.json',
    help='model config file')

args = parser.parse_args()
with open(args.config_file) as fin:
    config = json.load(fin, object_hook=lambda d: SimpleNamespace(**d))

bert_config = AutoConfig.from_pretrained(config.bert_model_path, cache_dir=TEMP)

WRAPPED_MODEL = AutoModelWithLMHead.from_pretrained(
            config.bert_model_path,
            from_tf=False,
            config=bert_config,
            cache_dir=TEMP,
        )

tokenizer = BertTokenizer.from_pretrained(config.bert_model_path)
WRAPPED_MODEL.resize_token_embeddings(len(tokenizer))
block_size = tokenizer.max_len

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=config.train_file_path,
    block_size=block_size,
)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

"""### Finally, we are all set to initialize our Trainer"""



training_args = TrainingArguments(
    output_dir=TEMP,
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=64,
    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=WRAPPED_MODEL,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
    prediction_loss_only=True,
)


trainer.train(model_path=config.bert_model_path)
trainer.save_model(output_dir=config.trained_model_path)


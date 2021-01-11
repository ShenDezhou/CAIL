"""Training file for SMP-CAIL2020-Argmine.

Author: Tsinghuaboy tsinghua9boy@sina.com

Usage:
    python -m torch.distributed.launch train.py \
        --config_file 'config/bert_config.json'
    CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch train.py \
        --config_file 'config/rnn_config.json'
"""
import itertools
from typing import Dict
import argparse
import json
import os
from copy import deepcopy
from types import SimpleNamespace

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from transformers.optimization import (
    AdamW, get_linear_schedule_with_warmup, get_constant_schedule)

from data import Data
from evaluate import evaluate, calculate_accuracy_f1, get_labels_from_file,handy_tool
from model import RnnForSentencePairClassification, BERNet, NERNet, NERWNet
from utils import get_csv_logger, get_path
from vocab import build_vocab


MODEL_MAP = {
    'bert': BERNet,
    'rnn': NERNet,
    'rnnkv': NERWNet
}


class Trainer:
    """Trainer for SMP-CAIL2020-Argmine.


    """
    def __init__(self,
                 model, data_loader: Dict[str, DataLoader], device, config):
        """Initialize trainer with model, data, device, and config.
        Initialize optimizer, scheduler, criterion.

        Args:
            model: model to be evaluated
            data_loader: dict of torch.utils.data.DataLoader
            device: torch.device('cuda') or torch.device('cpu')
            config:
                config.experiment_name: experiment name
                config.model_type: 'bert' or 'rnn'
                config.lr: learning rate for optimizer
                config.num_epoch: epoch number
                config.num_warmup_steps: warm-up steps number
                config.gradient_accumulation_steps: gradient accumulation steps
                config.max_grad_norm: max gradient norm

        """
        self.model = model
        self.device = device
        self.config = config
        self.data_loader = data_loader
        self.config.num_training_steps = config.num_epoch * (
            len(data_loader['train']) // config.batch_size)
        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_scheduler()
        self.criterion = nn.MultiLabelSoftMarginLoss(reduction='mean')

    def _get_optimizer(self):
        """Get optimizer for different models.

        Returns:
            optimizer
        """
        if self.config.model_type == 'bert':
            no_decay = ['bias', 'gamma', 'beta']
            optimizer_parameters = [
                {'params': [p for n, p in self.model.named_parameters()
                            if not any(nd in n for nd in no_decay)],
                 'weight_decay_rate': 0.01},
                {'params': [p for n, p in self.model.named_parameters()
                            if any(nd in n for nd in no_decay)],
                 'weight_decay_rate': 0.0}]
            optimizer = AdamW(
                optimizer_parameters,
                lr=self.config.lr,
                betas=(0.9, 0.999),
                weight_decay=1e-8,
                correct_bias=False)
        else:  # rnn
            optimizer = Adam(self.model.parameters(), lr=self.config.lr)
        return optimizer

    def _get_scheduler(self):
        """Get scheduler for different models.

        Returns:
            scheduler
        """
        if self.config.model_type == 'bert':
            scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config.num_warmup_steps,
                num_training_steps=self.config.num_training_steps)
        else:  # rnn
            scheduler = get_constant_schedule(self.optimizer)
        return scheduler

    def flatten(self, ll):
        return list(itertools.chain(*ll))

    def _evaluate_for_train_valid(self):
        """Evaluate model on train and valid set and get acc and f1 score.

        Returns:
            train_acc, train_f1, valid_acc, valid_f1
        """
        train_predictions, train_length = evaluate(
            model=self.model, data_loader=self.data_loader['valid_train'],
            device=self.device)
        valid_predictions, valid_length = evaluate(
            model=self.model, data_loader=self.data_loader['valid_valid'],
            device=self.device)

        train_answers = handy_tool(self.data_loader['train_label'], train_length)#get_labels_from_file(self.config.train_file_path)
        valid_answers = handy_tool(self.data_loader['valid_label'], valid_length)#get_labels_from_file(self.config.valid_file_path)
        train_predictions, valid_predictions = self.flatten(train_predictions), self.flatten(valid_predictions)
        train_answers, valid_answers = self.flatten(train_answers), self.flatten(valid_answers)
        train_acc, train_f1 = calculate_accuracy_f1(
            train_answers, train_predictions)
        valid_acc, valid_f1 = calculate_accuracy_f1(
            valid_answers, valid_predictions)
        return train_acc, train_f1, valid_acc, valid_f1

    def _epoch_evaluate_update_description_log(
            self, tqdm_obj, logger, epoch):
        """Evaluate model and update logs for epoch.

        Args:
            tqdm_obj: tqdm/trange object with description to be updated
            logger: logging.logger
            epoch: int

        Return:
            train_acc, train_f1, valid_acc, valid_f1
        """
        # Evaluate model for train and valid set
        results = self._evaluate_for_train_valid()
        train_acc, train_f1, valid_acc, valid_f1 = results
        # Update tqdm description for command line
        tqdm_obj.set_description(
            'Epoch: {:d}, train_acc: {:.6f}, train_f1: {:.6f}, '
            'valid_acc: {:.6f}, valid_f1: {:.6f}, '.format(
                epoch, train_acc, train_f1, valid_acc, valid_f1))
        # Logging
        logger.info(','.join([str(epoch)] + [str(s) for s in results]))
        return train_acc, train_f1, valid_acc, valid_f1

    def save_model(self, filename):
        """Save model to file.

        Args:
            filename: file name
        """
        torch.save(self.model.state_dict(), filename)

    def train(self):
        """Train model on train set and evaluate on train and valid set.

        Returns:
            state dict of the best model with highest valid f1 score
        """
        epoch_logger = get_csv_logger(
            os.path.join(self.config.log_path,
                         self.config.experiment_name + '-epoch.csv'),
            title='epoch,train_acc,train_f1,valid_acc,valid_f1')
        step_logger = get_csv_logger(
            os.path.join(self.config.log_path,
                         self.config.experiment_name + '-step.csv'),
            title='step,loss')
        trange_obj = trange(self.config.num_epoch, desc='Epoch', ncols=120)
        # self._epoch_evaluate_update_description_log(
        #     tqdm_obj=trange_obj, logger=epoch_logger, epoch=0)
        best_model_state_dict, best_train_f1, global_step = None, 0, 0
        for epoch, _ in enumerate(trange_obj):
            self.model.train()
            tqdm_obj = tqdm(self.data_loader['train'], ncols=80)
            for step, batch in enumerate(tqdm_obj):
                batch = tuple(t.to(self.device) for t in batch)
                # logits = self.model(*batch[:-1])  # the last one is label
                # loss = self.criterion(logits, batch[-1])
                loss = self.model(*batch)  # the last one is label

                # if self.config.gradient_accumulation_steps > 1:
                #     loss = loss / self.config.gradient_accumulation_steps
                # self.optimizer.zero_grad()
                loss.backward()

                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm)
                    #after 梯度累加的基本思想在于，在优化器更新参数前，也就是执行 optimizer.step() 前，进行多次反向传播，是的梯度累计值自动保存在 parameter.grad 中，最后使用累加的梯度进行参数更新。
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1
                    tqdm_obj.set_description('loss: {:.6f}'.format(loss.item()))
                    step_logger.info(str(global_step) + ',' + str(loss.item()))

            # if epoch >= 2:
            results = self._epoch_evaluate_update_description_log(
                tqdm_obj=trange_obj, logger=epoch_logger, epoch=epoch + 1)

            self.save_model(os.path.join(
                self.config.model_path, self.config.experiment_name,
                self.config.model_type + '-' + str(epoch + 1) + '.bin'))

            if results[-3] > best_train_f1:
                best_model_state_dict = deepcopy(self.model.state_dict())
                best_train_f1 = results[-3]

        return best_model_state_dict


def main(config_file='config/bert_config.json'):
    """Main method for training.

    Args:
        config_file: in config dir
    """
    # 0. Load config and mkdir
    with open(config_file) as fin:
        config = json.load(fin, object_hook=lambda d: SimpleNamespace(**d))
    get_path(os.path.join(config.model_path, config.experiment_name))
    get_path(config.log_path)
    # if config.model_type == 'rnn':  # build vocab for rnn
    #     build_vocab(file_in=config.all_train_file_path,
    #                 file_out=os.path.join(config.model_path, 'vocab.txt'))
    # 1. Load data
    data = Data(vocab_file=os.path.join(config.model_path, 'vocab.txt'),
                max_seq_len=config.max_seq_len,
                model_type=config.model_type, config=config)
    datasets = data.load_train_and_valid_files(
        train_file=config.train_file_path,
        valid_file=config.valid_file_path)
    train_set, valid_set_train, valid_set_valid, train_label, valid_label = datasets
    if torch.cuda.is_available():
        device = torch.device('cuda')
        # device = torch.device('cpu')
        # torch.distributed.init_process_group(backend="nccl")
        # sampler_train = DistributedSampler(train_set)
        sampler_train = RandomSampler(train_set)
    else:
        device = torch.device('cpu')
        sampler_train = RandomSampler(train_set)
    data_loader = {
        'train': DataLoader(
            train_set, sampler=sampler_train, batch_size=config.batch_size),
        'valid_train': DataLoader(
            valid_set_train, batch_size=config.batch_size, shuffle=False),
        'valid_valid': DataLoader(
            valid_set_valid, batch_size=config.batch_size, shuffle=False),
        "train_label":train_label,
        "valid_label":valid_label
    }
    # 2. Build model
    model = MODEL_MAP[config.model_type](config)
    #load model states.
    if config.trained_weight:
        model.load_state_dict(torch.load(config.trained_weight))
    model.to(device)
    if torch.cuda.is_available():
        model = model
        # model = torch.nn.parallel.DistributedDataParallel(
        #     model, find_unused_parameters=True)
    # 3. Train
    trainer = Trainer(model=model, data_loader=data_loader,
                      device=device, config=config)
    best_model_state_dict = trainer.train()
    # 4. Save model
    torch.save(best_model_state_dict,
               os.path.join(config.model_path, 'model.bin'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config_file', default='config/bert_config.json',
        help='model config file')

    parser.add_argument(
        '--local_rank', default=0,
        help='used for distributed parallel')
    args = parser.parse_args()
    main(args.config_file)

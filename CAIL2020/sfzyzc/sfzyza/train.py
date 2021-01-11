"""Training file for SMP-CAIL2020-Argmine.

Author: Tsinghuaboy tsinghua9boy@sina.com

Usage:
    python -m torch.distributed.launch train.py \
        --config_file 'config/bert_config.json'
    CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch train.py \
        --config_file 'config/rnn_config.json'
"""
import math
import time
from typing import Dict
import argparse
import json
import os
from copy import deepcopy
from types import SimpleNamespace

import numpy
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from transformers.optimization import (
    AdamW, get_linear_schedule_with_warmup, get_constant_schedule)

from data import Data
from evaluate import evaluate, calculate_accuracy_f1, get_labels_from_file
from model import BertForClassification
from utils import get_csv_logger, get_path
from vocab import build_vocab


import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.utils.utils as xu


MODEL_MAP = {
    'bert': BertForClassification
}
# Define Parameters

# SERIAL_EXEC = xmp.MpSerialExecutor()
# Only instantiate model weights once in memory.
#WRAPPED_MODEL = None#xmp.MpModelWrapper(ResNet18())

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
        self.criterion = nn.BCEWithLogitsLoss(reduction='mean')

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

    def _evaluate_for_train_valid(self):
        """Evaluate model on train and valid set and get acc and f1 score.

        Returns:
            train_acc, train_f1, valid_acc, valid_f1
        """
        train_predictions = evaluate(
            model=self.model, data_loader=self.data_loader['valid_train'],
            device=self.device)
        valid_predictions = evaluate(
            model=self.model, data_loader=self.data_loader['valid_valid'],
            device=self.device)
        train_answers = get_labels_from_file(self.config.train_file_path)
        valid_answers = get_labels_from_file(self.config.valid_file_path)
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
        best_model_state_dict, best_valid_f1, global_step = None, 0, 0
        for epoch, _ in enumerate(trange_obj):
            self.model.train()
            tqdm_obj = tqdm(self.data_loader['train'], ncols=80)
            for step, batch in enumerate(tqdm_obj):
                # batch = tuple(t.to(self.device) for t in batch)
                logits = self.model(*batch[:-1])  # the last one is label
                loss = self.criterion(logits, batch[-1])

                # if self.config.gradient_accumulation_steps > 1:
                #     loss = loss / self.config.gradient_accumulation_steps
                # self.optimizer.zero_grad()
                loss.backward()

                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.max_grad_norm)

                    # after 梯度累加的基本思想在于，在优化器更新参数前，也就是执行 optimizer.step() 前，进行多次反向传播，是的梯度累计值自动保存在 parameter.grad 中，最后使用累加的梯度进行参数更新。
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1
                    tqdm_obj.set_description('loss: {:.6f}'.format(loss.item()))
                    step_logger.info(str(global_step) + ',' + str(loss.item()))


            results = self._epoch_evaluate_update_description_log(
                tqdm_obj=trange_obj, logger=epoch_logger, epoch=epoch + 1)
            self.save_model(os.path.join(
                self.config.model_path, self.config.experiment_name,
                self.config.model_type + '-' + str(epoch + 1) + '.bin'))

            if results[-1] > best_valid_f1:
                best_model_state_dict = deepcopy(self.model.state_dict())
                best_valid_f1 = results[-1]
        return best_model_state_dict





def main(config_file='config/bert_config.json'):
    """Main method for training.

    Args:
        config_file: in config dir
    """
    global datasets
    # 0. Load config and mkdir
    with open(config_file) as fin:
        config = json.load(fin, object_hook=lambda d: SimpleNamespace(**d))


    get_path(os.path.join(config.model_path, config.experiment_name))
    get_path(config.log_path)
    if config.model_type in ['rnn', 'lr','cnn']:  # build vocab for rnn
        build_vocab(file_in=config.all_train_file_path,
                    file_out=os.path.join(config.model_path, 'vocab.txt'))
    # 1. Load data
    data = Data(vocab_file=os.path.join(config.model_path, 'vocab.txt'),
                max_seq_len=config.max_seq_len,
                model_type=config.model_type, config=config)


    def load_dataset():
        datasets = data.load_train_and_valid_files(
            train_file=config.train_file_path,
            valid_file=config.valid_file_path)
        return datasets

    if config.serial_load:
        datasets = SERIAL_EXEC.run(load_dataset)
    else:
        datasets = load_dataset()

    train_set, valid_set_train, valid_set_valid = datasets
    if torch.cuda.is_available():
        device = torch.device('cuda')
        # device = torch.device('cpu')
        # torch.distributed.init_process_group(backend="nccl")
        # sampler_train = DistributedSampler(train_set)
        sampler_train = RandomSampler(train_set)
    else:
        device = torch.device('cpu')
        sampler_train = RandomSampler(train_set)
    # TPU
    device = xm.xla_device()
    sampler_train = torch.utils.data.distributed.DistributedSampler(
        train_set,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=True)

    data_loader = {
        'train': DataLoader(
            train_set, sampler=sampler_train, batch_size=config.batch_size),
        'valid_train': DataLoader(
            valid_set_train, batch_size=config.batch_size, shuffle=False),
        'valid_valid': DataLoader(
            valid_set_valid, batch_size=config.batch_size, shuffle=False)}


    # 2. Build model
    # model = MODEL_MAP[config.model_type](config)
    model = WRAPPED_MODEL
    #load model states.
    # if config.trained_weight:
    #     model.load_state_dict(torch.load(config.trained_weight))
    model.to(device)
    if torch.cuda.is_available():
        model = model
        # model = torch.nn.parallel.DistributedDataParallel(
        #     model, find_unused_parameters=True)

    # 3. Train
    trainer = Trainer(model=model, data_loader=data_loader,
                      device=device, config=config)
    # best_model_state_dict = trainer.train()

    if config.model_type == 'bert':
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_parameters = [
            {'params': [p for n, p in model.named_parameters()
                        if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in model.named_parameters()
                        if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}]
        optimizer = AdamW(
            optimizer_parameters,
            lr=config.lr,
            betas=(0.9, 0.999),
            weight_decay=1e-8,
            correct_bias=False)
    else:  # rnn
        optimizer = Adam(model.parameters(), lr=config.lr)

    # if config.model_type == 'bert':
    #     scheduler = get_linear_schedule_with_warmup(
    #         optimizer,
    #         num_warmup_steps=config.num_warmup_steps,
    #         num_training_steps=config.num_training_steps)
    # else:  # rnn
    #     scheduler = get_constant_schedule(optimizer)

    criterion = nn.BCEWithLogitsLoss(reduction='mean')

    def train_loop_fn(loader):
        tracker = xm.RateTracker()
        model.train()
        for x, batch in enumerate(loader):
            # batch = tuple(t.to(self.device) for t in batch)
            output = model(*batch[:-1])  # the last one is label
            loss = criterion(output, batch[-1])
            loss.backward()
            # xm.optimizer_step(optimizer)
            # optimizer.zero_grad()

            tracker.add(FLAGS.batch_size)
            if (x + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.max_grad_norm)
                # after 梯度累加的基本思想在于，在优化器更新参数前，也就是执行 optimizer.step() 前，进行多次反向传播，是的梯度累计值自动保存在 parameter.grad 中，最后使用累加的梯度进行参数更新。
                xm.optimizer_step(optimizer)
                optimizer.zero_grad()

            if xm.get_ordinal() == 0:
                if x % FLAGS.log_steps == 0:
                    print('[xla:{}]({}) Loss={:.5f} Rate={:.2f} GlobalRate={:.2f} Time={}'.format(
                        xm.get_ordinal(), x, loss.item(), tracker.rate(),
                        tracker.global_rate(), time.asctime()), flush=True)

    def test_loop_fn(loader):
        total_samples = 0
        correct = 0
        model.eval()
        data, pred, target = None, None, None
        tracker = xm.RateTracker()
        for x, batch in enumerate(loader):
            output = model(*batch[:-1])  # the last one is label
            target = batch[-1]
            # pred = output.max(1, keepdim=True)[1]
            # correct += pred.eq(target.view_as(pred)).sum().item()
            for i in range(len(output)):
                logits = output[i].data.cpu().numpy().round()
                gold = target[i].data.cpu().numpy()
                correct += np.sum(logits == gold)
                total_samples += len(target[i])

            if xm.get_ordinal() == 0:
                if x % FLAGS.log_steps == 0:
                    print('[xla:{}]({}) Acc={:.5f} Rate={:.2f} GlobalRate={:.2f} Time={}'.format(
                        xm.get_ordinal(), x, correct*1.0/total_samples, tracker.rate(),
                        tracker.global_rate(), time.asctime()), flush=True)

        accuracy = 100.0 * correct / total_samples
        if xm.get_ordinal() == 0:
            print('[xla:{}] Accuracy={:.2f}%'.format(xm.get_ordinal(), accuracy), flush=True)
        return accuracy, data, pred, target

    # Train and eval loops
    accuracy = 0.0
    data, pred, target = None, None, None
    for epoch in range(FLAGS.num_epoch):
        para_loader = pl.ParallelLoader(data_loader['train'], [device])
        train_loop_fn(para_loader.per_device_loader(device))
        xm.master_print("Finished training epoch {}".format(epoch))

        # para_loader = pl.ParallelLoader(data_loader['valid_train'], [device])
        # accuracy_train, data, pred, target = test_loop_fn(para_loader.per_device_loader(device))

        para_loader = pl.ParallelLoader(data_loader['valid_valid'], [device])
        accuracy_valid, data, pred, target = test_loop_fn(para_loader.per_device_loader(device))
        xm.master_print("Finished test epoch {}, valid={:.2f}".format(epoch, accuracy_valid))

        if FLAGS.metrics_debug:
            xm.master_print(met.metrics_report())

        # 4. Save model
        # if xm.get_ordinal() == 0:
        #     # if epoch==FLAGS.num_epoch-1:
        #     # WRAPPED_MODEL.to('cpu')
        #     torch.save(WRAPPED_MODEL.state_dict(), os.path.join(
        #         config.model_path, config.experiment_name,
        #         config.model_type + '-' + str(epoch + 1) + '.bin'))
        #     xm.master_print('saved model.')
            # WRAPPED_MODEL.to(device)

    return accuracy_valid
    # 4. Save model
    # torch.save(best_model_state_dict,
    #            os.path.join(config.model_path, 'model.bin'))




def _mp_fn(rank, flags, model,serial):
    global WRAPPED_MODEL, FLAGS, SERIAL_EXEC
    WRAPPED_MODEL = model
    FLAGS = flags
    SERIAL_EXEC = serial

    accuracy_valid = main(args.config_file)
    # Retrieve tensors that are on TPU core 0 and plot.
    # plot_results(data.cpu(), pred.cpu(), target.cpu())
    xm.master_print(('DONE',  accuracy_valid))
    # 4. Save model
    if rank == 0:
        WRAPPED_MODEL.to('cpu')
        torch.save(WRAPPED_MODEL.state_dict(), os.path.join(config.model_path, 'model.bin'))
        xm.master_print('saved model.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config_file', default='config/bert_config.json',
        help='model config file')

    parser.add_argument(
        '--local_rank', default=0,
        help='used for distributed parallel')
    args = parser.parse_args()


    with open(args.config_file) as fin:
        config = json.load(fin, object_hook=lambda d: SimpleNamespace(**d))
    WRAPPED_MODEL = MODEL_MAP[config.model_type](config)
    if config.trained_weight:
        WRAPPED_MODEL.load_state_dict(torch.load(config.trained_weight))
    FLAGS = config
    SERIAL_EXEC = xmp.MpSerialExecutor()

    # main(args.config_file)
    xmp.spawn(_mp_fn, args=(FLAGS,WRAPPED_MODEL,SERIAL_EXEC,), nprocs=config.num_cores, start_method='fork')

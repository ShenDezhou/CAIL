import argparse
import gzip
import os
import pickle
from os.path import join
from tqdm import tqdm
from transformers import BertModel, BertTokenizer
from transformers import BertConfig as BC
import json

import torch
from torch import nn
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from model import *
from tools.utils import convert_to_tokens
from data_iterator_pack import IGNORE_INDEX
import numpy as np
import queue
import random
from config import set_config
from data_helper import DataHelper
from data_process import InputFeatures,Example


try:
    from apex import amp
except Exception:
    print('Apex not import!')

from data_process import read_examples, convert_examples_to_features
from evaluate.evaluate import eval
from utils import get_path,get_csv_logger

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def dispatch(context_encoding, context_mask, batch, device):
    batch['context_encoding'] = context_encoding.cuda(device)
    batch['context_mask'] = context_mask.float().cuda(device)
    return batch


def compute_loss(batch, start_logits, end_logits, type_logits, sp_logits, start_position, end_position):
    loss1 = criterion(start_logits, batch['y1']) + criterion(end_logits, batch['y2'])
    loss2 = args.type_lambda * criterion(type_logits, batch['q_type'])
    sent_num_in_batch = batch["start_mapping"].sum()
    loss3 = args.sp_lambda * sp_loss_fct(sp_logits.view(-1), batch['is_support'].float().view(-1)).sum() / sent_num_in_batch
    loss = loss1 + loss2 + loss3
    return loss, loss1, loss2, loss3


@torch.no_grad()
def predict(model, dataloader, example_dict, feature_dict, prediction_file, need_sp_logit_file=False):

    model.eval()
    answer_dict = {}
    sp_dict = {}
    dataloader.refresh()
    total_test_loss = [0] * 5

    for batch in tqdm(dataloader):

        batch['context_mask'] = batch['context_mask'].float()
        start_logits, end_logits, type_logits, sp_logits, start_position, end_position = model(batch)

        loss_list = compute_loss(batch, start_logits, end_logits, type_logits, sp_logits, start_position, end_position)

        for i, l in enumerate(loss_list):
            if not isinstance(l, int):
                total_test_loss[i] += l.item()


        answer_dict_ = convert_to_tokens(example_dict, feature_dict, batch['ids'], start_position.data.cpu().numpy().tolist(),
                                         end_position.data.cpu().numpy().tolist(), np.argmax(type_logits.data.cpu().numpy(), 1))
        answer_dict.update(answer_dict_)

        predict_support_np = torch.sigmoid(sp_logits).data.cpu().numpy()
        for i in range(predict_support_np.shape[0]):
            cur_sp_pred = []
            cur_id = batch['ids'][i]

            cur_sp_logit_pred = []  # for sp logit output
            for j in range(predict_support_np.shape[1]):
                if j >= len(example_dict[cur_id].sent_names):
                    break
                if need_sp_logit_file:
                    temp_title, temp_id = example_dict[cur_id].sent_names[j]
                    cur_sp_logit_pred.append((temp_title, temp_id, predict_support_np[i, j]))
                if predict_support_np[i, j] > args.sp_threshold:
                    cur_sp_pred.append(example_dict[cur_id].sent_names[j])
            sp_dict.update({cur_id: cur_sp_pred})

    new_answer_dict={}
    for key,value in answer_dict.items():
        new_answer_dict[key]=value.replace(" ","")
    prediction = {'answer': new_answer_dict, 'sp': sp_dict}
    with open(prediction_file, 'w',encoding='utf8') as f:
        json.dump(prediction, f,indent=4,ensure_ascii=False)

    for i, l in enumerate(total_test_loss):
        print("Test Loss{}: {}".format(i, l / len(dataloader)))
    test_loss_record.append(sum(total_test_loss[:3]) / len(dataloader))


def train_epoch(data_loader, model, logger, predict_during_train=False, epoch=1):
    model.train()
    pbar = tqdm(total=len(data_loader))
    epoch_len = len(data_loader)
    step_count = 0
    predict_step = epoch_len // 2
    while not data_loader.empty():
        step_count += 1
        batch = next(iter(data_loader))
        batch['context_mask'] = batch['context_mask'].float()
        train_batch(model, batch)
        # del batch
        if predict_during_train and (step_count % predict_step == 0):
            predict(model, eval_dataset, dev_example_dict, dev_feature_dict,
                     join(args.prediction_path, 'pred_seed_{}_epoch_{}_{}.json'.format(args.seed, epc, step_count)))
            eval(join(args.prediction_path, 'pred_seed_{}_epoch_{}_{}.json'.format(args.seed, epc, step_count)), args.validdata)
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), join(args.checkpoint_path, "ckpt_seed_{}_epoch_{}_{}.pkl".format(args.seed, epc, step_count)))
            model.train()
        pbar.update(1)

    predict(model, eval_dataset, dev_example_dict, dev_feature_dict,
             join(args.prediction_path, 'pred_seed_{}_epoch_{}_99999.json'.format(args.seed, epc)))
    results = eval(join(args.prediction_path, 'pred_seed_{}_epoch_{}_99999.json'.format(args.seed, epc)), args.validdata)
    # Logging
    keys='em,f1,prec,recall,sp_em,sp_f1,sp_prec,sp_recall,joint_em,joint_f1,joint_prec,joint_recall'.split(',')
    logger.info(','.join([str(epoch)] + [str(results[s]) for s in keys]))
    model_to_save = model.module if hasattr(model, 'module') else model
    torch.save(model_to_save.state_dict(), join(args.checkpoint_path, "ckpt_seed_{}_epoch_{}_99999.pkl".format(args.seed, epc)))


def train_batch(model, batch):
    global global_step, total_train_loss

    start_logits, end_logits, type_logits, sp_logits, start_position, end_position = model(batch)
    loss_list = compute_loss(batch, start_logits, end_logits, type_logits, sp_logits, start_position, end_position)
    loss_list = list(loss_list)
    if args.gradient_accumulation_steps > 1:
        # loss_list[0] = loss_list[0] / args.gradient_accumulation_steps
        loss_list[0] /= args.gradient_accumulation_steps
    
    if args.fp16:
        with amp.scale_loss(loss_list[0], optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        # loss_list[0].backward()
        loss_list[0].backward()

    if (global_step + 1) % args.gradient_accumulation_steps == 0:
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    global_step += 1

    for i, l in enumerate(loss_list):
        if not isinstance(l, int):
            total_train_loss[i] += l.item()

    if global_step % VERBOSE_STEP == 0:
        print("{} -- In Epoch{}: ".format(args.name, epc))
        for i, l in enumerate(total_train_loss):
            print("Avg-LOSS{}/batch/step: {}".format(i, l / VERBOSE_STEP))
        total_train_loss = [0] * 5


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = set_config()
    get_path("log/")

    args.n_gpu = torch.cuda.device_count()

    if args.seed == 0:
        args.seed = random.randint(0, 100)
        set_seed(args)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model)
    examples = read_examples(full_file=args.rawdata)
    with gzip.open("data_model/train_example.pkl.gz", 'wb') as fout:
        pickle.dump(examples, fout)

    features = convert_examples_to_features(examples, tokenizer, max_seq_length=args.max_seq_len, max_query_length=args.max_query_len)
    with gzip.open("data_model/train_feature.pkl.gz", 'wb') as fout:
        pickle.dump(features, fout)

    # tokenizer = BertTokenizer.from_pretrained(args.bert_model)
    examples = read_examples(full_file=args.validdata)
    with gzip.open("data_model/dev_example.pkl.gz", 'wb') as fout:
        pickle.dump(examples, fout)

    features = convert_examples_to_features(examples, tokenizer, max_seq_length=args.max_seq_len, max_query_length=args.max_query_len)
    with gzip.open("data_model/dev_feature.pkl.gz", 'wb') as fout:
        pickle.dump(features, fout)

    helper = DataHelper(gz=True, config=args)
    args.n_type = helper.n_type  # 2

    # Set datasets
    Full_Loader = helper.train_loader
    # Subset_Loader = helper.train_sub_loader
    dev_example_dict = helper.dev_example_dict
    dev_feature_dict = helper.dev_feature_dict
    eval_dataset = helper.dev_loader



    # roberta_config = BC.from_pretrained(args.bert_model)
    encoder = BertModel.from_pretrained(args.bert_model)
    # args.input_dim=roberta_config.hidden_size
    model = BertSupportNet(config=args, encoder=encoder)
    if args.trained_weight:
        model.load_state_dict(torch.load(args.trained_weight))
    model.to('cuda')

    # Initialize optimizer and criterions
    lr = args.lr
    t_total = len(Full_Loader) * args.epochs // args.gradient_accumulation_steps
    warmup_steps = args.warmup_step
    optimizer = AdamW(model.parameters(), lr=lr, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=t_total)
    criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=IGNORE_INDEX)  # 交叉熵损失
    binary_criterion = nn.BCEWithLogitsLoss(reduction='mean')  # 二元损失
    sp_loss_fct = nn.BCEWithLogitsLoss(reduction='none')  # 用于sp，平均值自己算

    if args.fp16:
        import apex
        apex.amp.register_half_function(torch, "einsum")
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    model = torch.nn.DataParallel(model)
    model.train()

    # Training
    global_step = epc = 0
    total_train_loss = [0] * 5
    test_loss_record = []
    VERBOSE_STEP = args.verbose_step

    epoch_logger = get_csv_logger(os.path.join("log/", args.name + '-epoch.csv'),
        title='epoch,em,f1,prec,recall,sp_em,sp_f1,sp_prec,sp_recall,joint_em,joint_f1,joint_prec,joint_recall')
    while True:
        if epc == args.epochs:  # 5 + 30
            exit(0)
        epc += 1

        Loader = Full_Loader
        Loader.refresh()

        train_epoch(Loader, model, logger=epoch_logger, predict_during_train=False, epoch=epc)
        # if epc > 2:
        #
        # else:
        #     train_epoch(Loader, model, logger=None)

import argparse
import os
import json
from os.path import join


def process_arguments(args):
    args.checkpoint_path = join(args.checkpoint_path, args.name)
    args.prediction_path = join(args.prediction_path, args.name)
    args.max_query_len = 50
    args.max_doc_len = 512


def save_settings(args):
    os.makedirs(args.checkpoint_path, exist_ok=True)
    os.makedirs(args.prediction_path, exist_ok=True)
    json.dump(args.__dict__, open(join(args.checkpoint_path, "run_settings.json"), 'w'))


def set_config():
    parser = argparse.ArgumentParser()
    data_path = 'output'

    # Required parameters
    parser.add_argument("--name", type=str, default='default')
    parser.add_argument("--prediction_path", type=str, default=join(data_path, 'submissions'))
    parser.add_argument("--checkpoint_path", type=str, default=join(data_path, 'checkpoints'))
    parser.add_argument("--data_dir", type=str, default='data_model')
    parser.add_argument("--rawdata", type=str, default='data2/trainx.json')
    parser.add_argument("--validdata", type=str, default='data2/valid.json')
    parser.add_argument("--output", type=str, default='result.txt')

    parser.add_argument("--fp16", action='store_true')

    parser.add_argument("--ckpt_id", type=int, default=0)
    parser.add_argument("--bert_model", type=str, default=r'D:\lbwj\model\bert\roberta_wwm_ext',
                        help='Currently only support bert-base-uncased and bert-large-uncased')

    # learning and log
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--qat_epochs", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_seq_len", type=int, default=512)

    #cnn
    parser.add_argument("--cnn_hidden_size", type=int, default=512)
    parser.add_argument("--cnn_output_size", type=int, default=512)
    parser.add_argument("--fc_hidden_size", type=int, default=64)
    # parser.add_argument("--cnn_module_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--resnet_type", type=int, default=8)

    parser.add_argument("--max_query_len", type=int, default=50)
    parser.add_argument("--max_bert_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument('--decay', type=float, default=0.99)
    parser.add_argument('--early_stop_epoch', type=int, default=0)
    parser.add_argument("--verbose_step", default=100000, type=int)
    parser.add_argument("--warmup_step", default=1000, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=64, type=int)
    parser.add_argument("--seed", default=0, type=int)

    parser.add_argument('--q_update', action='store_true', help='Whether update query')
    parser.add_argument("--prediction_trans", action='store_true', help='transformer version prediction layer')
    parser.add_argument("--trans_drop", type=float, default=0.5)
    parser.add_argument("--trans_heads", type=int, default=3)

    parser.add_argument("--input_dim", type=int, default=768, help="bert-base=768, bert-large=1024")

    parser.add_argument("--model_gpu", default='0', type=str, help="device to place model.")
    # parser.add_argument('--trained_weight',default=None)
    parser.add_argument('--trained_weight', default="")

    # loss
    parser.add_argument("--type_lambda", type=float, default=1)
    parser.add_argument("--sp_lambda", type=float, default=5)
    parser.add_argument("--sp_threshold", type=float, default=0.5)
    parser.add_argument('--label_type_num', default=4, type=int)#yes/no/unknown/span

    parser.add_argument('--num_cores', default=1, type=int)

    args = parser.parse_args()

    process_arguments(args)
    save_settings(args)

    return args

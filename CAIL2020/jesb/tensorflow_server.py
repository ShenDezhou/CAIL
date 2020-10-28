import argparse
import logging
import os
import time
from types import SimpleNamespace
import falcon
import pandas
import torch
from falcon_cors import CORS
import json
import waitress
import numpy as np
import tensorflow as tf
from tensorflow.contrib.crf import viterbi_decode

import json
import re
from data_utils import input_from_line, result_to_json
from six import unichr

logging.basicConfig(level=logging.INFO, format='%(asctime)-18s %(message)s')
logger = logging.getLogger()
cors_allow_all = CORS(allow_all_origins=True,
                      allow_origins_list=['*'],
                      allow_all_headers=True,
                      allow_all_methods=True,
                      allow_credentials_all_origins=True
                      )

parser = argparse.ArgumentParser()
parser.add_argument(
    '-j', '--json_file', default='model_data/money_maps.json',
    help='model config file')
parser.add_argument(
    '-p', '--pb_path', default='model_data/money_model.pb',
    help='model config file')
parser.add_argument(
    '-b', '--batch_size', default=1,
    help='model config file')
parser.add_argument(
    '-n', '--num_tags', default=4,
    help='model config file')


args = parser.parse_args()

json_file=args.json_file
pb_path = args.pb_path
batch_size=args.batch_size
num_tags=args.num_tags


def create_feed_dict(batch, char_inputs, seg_inputs, dropout):
    """
    :param batch: list train/evaluate data
    :return: structured data to feed
    """
    _, chars, segs, tags = batch
    feed_dict = {
        char_inputs: np.asarray(chars),
        seg_inputs: np.asarray(segs),
        dropout: 1.0,  # dropout = 1 if test
    }

    return feed_dict


def decode(logits, lengths, matrix):
        """
        Get the predicted path
        :param logits: [batch_size, num_steps, num_tags]float32, logits
        :param lengths: [batch_size]int32, real length of each sequence
        :param matrix: transaction matrix for inference
        :return:
        """
        # inference final labels use viterbi Algorithm
        paths = []
        small = -1000.0
        start = np.asarray([[small]*num_tags + [0]])
        for score, length in zip(logits, lengths):
            score = score[:length]
            pad = small * np.ones([length, 1])
            logits = np.concatenate([score, pad], axis=1)
            logits = np.concatenate([start, logits], axis=0)
            path, _ = viterbi_decode(logits, matrix)
            # viterbi_decode(score,transition_params)
            # 通俗一点,作用就是返回最好的标签序列.这个函数只能够在测试时使用,在tensorflow外部解码
            # 参数：
            # score: 一个形状为[seq_len, num_tags] matrix of unary potentials.
            # transition_params: 形状为[num_tags+1, num_tags+1] 的转移矩阵
            # 返回：
            # viterbi: 一个形状为[seq_len] 显示了最高分的标签索引的列表. 最佳路径
            # viterbi_score: A float containing the score for the Viterbi sequence.

            paths.append(path[1:])
        return paths

#digit regex[0-9]
digit_regex = re.compile("[0-9,]+")

def strQ2B(ustring):
    """全角转半角"""
    rstring = ""
    for uchar in ustring:
        inside_code=ord(uchar)
        if inside_code == 12288:#全角空格直接转换
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374): #全角字符（除空格）根据关系转化
            inside_code -= 65248

        rstring += unichr(inside_code)
    return rstring

def augment(word, line):
    line = strQ2B(line)
    augdigit = re.compile("([0-9]*"+word+")")
    augword = augdigit.search(line)
    #找到了多个匹配，不修改
    if augword and len(augword.groups()) == 1:
        return augword.group(0)
    return word


class TorchResource:

    def __init__(self):
        logger.info("...")
        with open(args.json_file, "r") as f:  # with open(FLAGS.map_file, "rb") as f:
            self.char_to_id, self.id_to_char, self.tag_to_id, self.id_to_tag = json.load(f)  # pickle.load(f)
            print('json file loaded')

        logger.info("###")




    def predict_from_pb(self, document):
        row = {'content': document}
        df = pandas.DataFrame().append(row, ignore_index=True)
        filename = "data/{}.csv".format(time.time())
        df.to_csv(filename, index=False, escapechar="\\", columns=['content'])

        with tf.Graph().as_default():
            output_graph_def = tf.GraphDef()
            with open(pb_path, "rb") as f:
                output_graph_def.ParseFromString(f.read())
                tf.import_graph_def(output_graph_def, name="")

            with tf.Session() as sess:  # config=config
                sess.run(tf.global_variables_initializer())

                # Get the input placeholders from the graph by name
                char_input = sess.graph.get_tensor_by_name('CharInputs:0')
                seg_input = sess.graph.get_tensor_by_name('SegInputs:0')
                drop_keep_prob = sess.graph.get_tensor_by_name('Dropout:0')
                # Tensors we want to evaluate,outputs
                lengths = sess.graph.get_tensor_by_name('lengths:0')
                logits = sess.graph.get_tensor_by_name("project/logits_outputs:0")
                # predictions = sess.graph.get_tensor_by_name("Accuracy/predictions:0")

                trans = sess.graph.get_tensor_by_name("crf_loss/transitions:0")

                # fo = open(test_data_path, "r", encoding='utf8')
                # all_data = fo.readlines()
                # fo.close()
                # for line in all_data:  # 一行行遍历
                lines = document.split(r"\n")
                lines = [line for line in lines if len(line)>0]


                list_amounts = []
                for line in lines:
                    input_batch = input_from_line(line, self.char_to_id)  # 处理测试数据格式
                    feed_dict = create_feed_dict(input_batch, char_input, seg_input, drop_keep_prob)  # 创建输入的feed_dict
                    seq_len, scores = sess.run([lengths, logits], feed_dict)
                    print('---')

                    transition_matrix = trans.eval()
                    batch_paths = decode(scores, seq_len, transition_matrix)
                    tags = [self.id_to_tag[str(idx)] for idx in batch_paths[0]]
                    print(tags)
                    result = result_to_json(input_batch[0][0], tags)
                    original = str(result['string'])
                    entities = result['entities']

                    if len(entities) != 0:
                        for entity in entities:
                            #是数字金额需要增强逻辑
                            if digit_regex.match(entity['word']) and len(entity['word']) >= 5:
                                aug_word = augment(entity['word'], line)
                                list_amounts.append(aug_word)
                            else:
                                list_amounts.append(entity['word'])
                            print(entity['word'])
                return {"answer":list_amounts}

    def on_get(self, req, resp):
        logger.info("...")
        resp.set_header('Access-Control-Allow-Origin', '*')
        resp.set_header('Access-Control-Allow-Methods', '*')
        resp.set_header('Access-Control-Allow-Headers', '*')
        resp.set_header('Access-Control-Allow-Credentials','true')
        content = req.get_param('1', True)
        # clean_content = cleanall(content)
        clean_content = content
        resp.media = self.predict_from_pb(clean_content)
        logger.info("###")


    def on_post(self, req, resp):
        """Handles POST requests"""
        resp.set_header('Access-Control-Allow-Origin', '*')
        resp.set_header('Access-Control-Allow-Methods', '*')
        resp.set_header('Access-Control-Allow-Headers', '*')
        resp.set_header('Access-Control-Allow-Credentials', 'true')
        resp.set_header("Cache-Control", "no-cache")
        data = req.stream.read(req.content_length)
        data = data.decode('utf-8')
        # regex = re.compile(r'\\(?![/u"])')
        # data = regex.sub(r"\\", data)
        jsondata = json.loads(data)
        # clean_title = shortenlines(jsondata['1'])
        # clean_content = cleanall(jsondata['2'])
        clean_content = jsondata['1']
        resp.media = self.predict_from_pb(clean_content)
        logger.info("###")

if __name__=="__main__":
    api = falcon.API(middleware=[cors_allow_all.middleware])
    api.req_options.auto_parse_form_urlencoded = True
    api.add_route('/z', TorchResource())
    waitress.serve(api, port=58084, threads=48, url_scheme='http')

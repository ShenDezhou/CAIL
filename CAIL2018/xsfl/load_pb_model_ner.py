import numpy as np
import tensorflow as tf
from tensorflow.contrib.crf import viterbi_decode

import json

from data_utils import input_from_line, result_to_json

# 添加命令行的optional argument（可选参数）
flags = tf.app.flags

flags.DEFINE_string("json_file", "model_data/money_maps.json", "file for json")  # shiyou_maps.json

FLAGS = tf.app.flags.FLAGS  # 从对应的命令行参数取出参数

pb_path = "model_data/money_model.pb"
batch_size = 1
test_data_path = "data/contract_amount_test.csv"
num_tags = 4


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


def predict_from_pb():
    with open(FLAGS.json_file, "r") as f:  # with open(FLAGS.map_file, "rb") as f:
        char_to_id, id_to_char, tag_to_id, id_to_tag = json.load(f)  # pickle.load(f)
        print('json file loaded')
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

            fo = open(test_data_path, "r", encoding='utf8')
            all_data = fo.readlines()
            fo.close()
            for line in all_data:  # 一行行遍历
                input_batch = input_from_line(line, char_to_id)  # 处理测试数据格式
                feed_dict = create_feed_dict(input_batch,char_input,seg_input,drop_keep_prob)  # 创建输入的feed_dict
                seq_len,scores = sess.run([lengths,logits], feed_dict)
                print('---')

                transition_matrix = trans.eval()
                batch_paths = decode(scores, seq_len, transition_matrix)
                tags = [id_to_tag[str(idx)] for idx in batch_paths[0]]
                print(tags)
                result = result_to_json(input_batch[0][0], tags)
                original = str(result['string'])
                entities = result['entities']
                if len(entities) != 0:
                    for entity in entities:
                        print(entity['word'])


            # print("总样本数:", len(all_data))
            # fw.close()


def main(_):
    predict_from_pb()
    # evaluate_from_excel()


if __name__ == "__main__":
    tf.app.run(main)

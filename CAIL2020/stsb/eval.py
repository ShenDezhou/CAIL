#!/usr/bin/env python
# coding=utf-8
"""
Entity Linking效果评估脚本，评价指标Micro-F1
Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved 
"""
import json
from collections import defaultdict


class Eval(object):
    """
    Entity Linking Evaluation
    """

    def __init__(self, golden_file_path, user_file_path):
        """
        Init Instance 

        Args:
            golden_file_path (str): golden 集合路径
            user_file_path (str): 用户预测结果路径
        """
        self.golden_file_path = golden_file_path
        self.user_file_path = user_file_path
        self.tp = 0
        self.fp = 0
        self.total_recall = 0
        self.errno = None

    def format_check(self, file_path):
        """
        文件格式验证
        :param file_path: 文件路径
        :return: Bool类型：是否通过格式检查，通过为True，反之False
        """
        flag = True
        
        # bad case 1
        if not file_path.endswith('.json'):
            raise Exception('The format of result should be .json')
        
        for line in open(file_path, encoding='utf-8'):
            json_info = json.loads(line.strip())
            
            # bad case 2
            if 'text_id' not in json_info:
                raise Exception(f"The 'text_id' is not the key of {json_info}"[:100])
            # bad case 3
            if 'text' not in json_info:
                raise Exception(f"The 'text' is not the key of {json_info}"[:100])
            # bad case 4
            if 'mention_data' not in json_info:
                raise Exception(f"The 'mention_data' is not the key of {json_info}"[:100])
            # bad case 5
            if not isinstance(json_info['text_id'], str):
                raise Exception(f"The 'text_id':{json_info['text_id']} is not str"[:100])
            # bad case 6
            if not json_info['text_id'].isdigit():
                raise Exception(f"The 'text_id':{json_info['text_id']} is not str of digit"[:100])
            # bad case 7
            if not isinstance(json_info['text'], str):
                raise Exception(f"The 'text':{json_info['text']} is not str"[:100])
            # bad case 8
            if not isinstance(json_info['mention_data'], list):
                raise Exception(f"The 'mention_data':{json_info['mention_data']} is not str"[:100])
            
            for mention_info in json_info['mention_data']:
                # bad case 9           
                if 'kb_id' not in mention_info:
                    raise Exception(f"The 'kb_id' is not the key of {mention_info}"[:100])
                # bad case 10
                if 'mention' not in mention_info:
                    raise Exception(f"The 'mention' is not the key of {mention_info}"[:100])
                # bad case 11
                if 'offset' not in mention_info:
                    raise Exception(f"The 'offset' is not the key of {mention_info}"[:100])
                # bad case 12
                # if not isinstance(mention_info['kb_id'], str):
                #     raise Exception(f"The 'kb_id':{mention_info['kb_id']} is not str"[:100])
                # bad case 13
                if not isinstance(mention_info['mention'], str):
                    raise Exception(f"The 'mention':{mention_info['mention']} is not str"[:100])
                # bad case 14
                if not isinstance(mention_info['offset'], str):
                    raise Exception(f"The 'offset':{mention_info['offset']} is not str"[:100])
                # bad case 15 
                if not mention_info['offset'].isdigit():
                    raise Exception(f"The 'offset':{mention_info['offset']} is not str of digit"[:100])


    def micro_f1(self):
        """
        :return: float类型：精确率，召回率，Micro-F1值
        """
        # 文本格式验证
        self.format_check(self.golden_file_path)
        self.format_check(self.user_file_path)
        
        precision = 0
        recall = 0
        self.tp = 0
        self.fp = 0
        self.total_recall = 0
        golden_dict = defaultdict(list)
        golden_text_list = []
        for line in open(self.golden_file_path, encoding='utf-8'):
            golden_info = json.loads(line.strip())
            text_id = golden_info['text_id']
            text = golden_info['text']
            mention_data = golden_info['mention_data']
            golden_text_list.append(text_id)
            for mention_info in mention_data:
                kb_id = str(mention_info['kb_id'])
                mention = mention_info['mention']
                offset = mention_info['offset']
                key = '\1'.join([text_id, text, mention, offset]).encode('utf8')
                # value的第二个元素表示标志位，用于判断是否已经进行了统计
                golden_dict[key] = [kb_id, 0]
                self.total_recall += 1

        # 进行评估
        golden_text_set = set(golden_text_list)
        for line in open(self.user_file_path, encoding='utf-8'):
            golden_info = json.loads(line.strip())
            text_id = golden_info['text_id']
            if text_id not in golden_text_set:
                continue
            text = golden_info['text']
            mention_data = golden_info['mention_data']
            for mention_info in mention_data:
                kb_id = str(mention_info['kb_id'])
                mention = mention_info['mention']
                offset = mention_info['offset']
                key = '\1'.join([text_id, text, mention, offset]).encode('utf8')
                if key in golden_dict:
                    kb_result_golden = golden_dict[key]
                    if kb_id.isdigit():
                        if kb_id in [kb_result_golden[0]] and kb_result_golden[1] in [0]:
                            self.tp += 1
                        else:
                            self.fp += 1
                    else:
                        # nil golden结果
                        nil_res = kb_result_golden[0].split('|')
                        if kb_id in nil_res and kb_result_golden[1] in [0]:
                            self.tp += 1
                        else:
                            self.fp += 1
                    golden_dict[key][1] = 1
                else:
                    self.fp += 1
        if self.tp + self.fp > 0:
            precision = float(self.tp) / (self.tp + self.fp)
        if self.total_recall > 0:
            recall = float(self.tp) / self.total_recall
        a = 2 * precision * recall
        b = precision + recall
        if b == 0:
            return 0, 0, 0
        f1 = a / b
        return dict(F1=f1,Precision = precision, Recall = recall)


def eval(golden_file_path, user_file_path):
    """
    [summary]

    Args:
        golden_file_path (str): golden 集合路径
        user_file_path (str): 用户预测结果路径

    Returns:
        dict: the key is [F1,Precision,Recall]
    """
    instance = Eval(golden_file_path, user_file_path)
    return instance.micro_f1()

if __name__ == '__main__':
    # import os
    # dirname, filename = os.path.split(os.path.abspath(__file__))
    # father_dirname, _ = os.path.split(dirname)
    print(eval('data/dev.json', 'data/result.json'))

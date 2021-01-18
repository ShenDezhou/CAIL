import argparse
import itertools
import logging
import os
import time
from types import SimpleNamespace
import falcon
import pandas
import torch
from falcon_cors import CORS
import waitress
import numpy as np

import json
import re
from torch.utils.data import DataLoader

from data import Data
from evaluate import evaluate, handy_tool, calculate_accuracy_f1
from model import BERNet, BERXLNet, NERNet, NERWNet
from utils import load_torch_model


MODEL_MAP = {
    'bert': BERNet,
    'bertxl': BERXLNet,
    'rnn': NERNet,
    'rnnkv': NERWNet
}


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
    '-p', '--port', default=58081,
    help='falcon server port')
parser.add_argument(
    '-c', '--config_file', default='config/bert_config-xl.json',
    help='model config file')
args = parser.parse_args()
model_config=args.config_file


def result_to_json(string, tags):
    item = {"string": string, "entities": []}
    entity_name = ""
    entity_start = 0
    idx = 0
    i = -1
    zipped = zip(string, tags)
    listzip = list(zipped)
    last = len(listzip)
    for char, tag in listzip:
        i += 1
        if tag == 3:
            item["entities"].append({"word": char, "start": idx, "end": idx+1, "type":'s'})
        elif tag == 0:
            entity_name += char
            entity_start = idx
        elif tag == 1:
            if (entity_name != "") and (i == last):
                entity_name += char
                item["entities"].append({"word": entity_name, "start": entity_start, "end": idx + 1, "type": 'bms'})
                entity_name = ""
            else:
                entity_name += char
        elif tag == 2:  # or i == len(zipped)
            entity_name += char
            item["entities"].append({"word": entity_name, "start": entity_start, "end": idx + 1, "type": 'bms'})
            entity_name = ""
        else:
            entity_name = ""
            entity_start = idx
        idx += 1
    return item



class TorchResource:

    def __init__(self):
        logger.info("...")
        # 0. Load config
        with open(model_config) as fin:
            self.config = json.load(fin, object_hook=lambda d: SimpleNamespace(**d))
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        # 1. Load data
        self.data = Data(vocab_file=os.path.join(self.config.model_path, 'vocab.txt'),
                    max_seq_len=self.config.max_seq_len,
                    model_type=self.config.model_type, config=self.config)

        # 2. Load model
        self.model = MODEL_MAP[self.config.model_type](self.config)
        self.model = load_torch_model(
            self.model, model_path=os.path.join(self.config.model_path, 'model.bin'))
        self.model.to(self.device)
        logger.info("###")

    def flatten(self, ll):
        return list(itertools.chain(*ll))

    def cleanall(self, content):
        return content.replace(" ", "", 10**10)

    def split(self, content):
        line = re.findall('(.*?(?:[\n ]|.$))', content)
        sublines = []
        for l in line:
            if len(l) > self.config.max_seq_len:
                ll = re.findall('(.*?(?:[。，]|.$))', l)
                sublines.extend(ll)
            else:
                sublines.append(l)
        sublines = [l for l in sublines if len(l.strip())> 0]
        return sublines

    def bert_classification(self, content):
        logger.info('1:{}'.format( content))
        lines = self.split(content)
        rows = []
        for line in lines:
            rows.append( {'content': line})
        df = pandas.DataFrame(rows)
        filename = "data/{}.csv".format(time.time())
        df.to_csv(filename, index=False, columns=['content'])
        test_set, sc_list, label_list, row_list = self.data.load_file(filename, train=False)

        # token_list = []
        # for line in sc_list:
        #     tokens = self.data.tokenizer.convert_ids_to_tokens(line)
        #     token_list.append(tokens)

        data_loader_test = DataLoader(
            test_set, batch_size=self.config.batch_size, shuffle=False)
        # Evaluate
        answer_list, length_list  = evaluate(self.model, data_loader_test, self.device, isTest=True)
        mod_tokens_list = handy_tool(row_list, length_list)
        result = [result_to_json(t, s) for t, s in zip(mod_tokens_list, answer_list)]
        entities = [item['entities'] for item in result]
        entities = self.flatten(entities)

        return {"data": entities}

    def on_get(self, req, resp):
        logger.info("...")
        resp.set_header('Access-Control-Allow-Origin', '*')
        resp.set_header('Access-Control-Allow-Methods', '*')
        resp.set_header('Access-Control-Allow-Headers', '*')
        resp.set_header('Access-Control-Allow-Credentials','true')
        content = req.get_param('text', True)
        # clean_content =
        #clean_content = self.cleanall(content)
        resp.media = self.bert_classification(content)
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
        content = jsondata['text']
        # clean_content = self.cleanall(content)
        resp.media = self.bert_classification(content)
        logger.info("###")

if __name__=="__main__":
    api = falcon.API(middleware=[cors_allow_all.middleware])
    api.req_options.auto_parse_form_urlencoded = True
    api.add_route('/z', TorchResource())
    waitress.serve(api, port=args.port, threads=48, url_scheme='http')

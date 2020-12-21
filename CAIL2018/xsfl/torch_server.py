import argparse
import logging
import os
import sys
from types import SimpleNamespace
import falcon
import pandas
import torch
from falcon_cors import CORS
import json
import waitress
from data import Data
from torch.utils.data import DataLoader
from utils import load_torch_model
from model import BertForClassification, RnnForSentencePairClassification, FullyConnectNet, LogisticRegression, CharCNN
from evaluate import evaluate
import time

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
    '-p', '--port', default=58073,
    help='falcon server port')
parser.add_argument(
    '-c', '--config_file', default='config/bert_config.json',
    help='model config file')
args = parser.parse_args()
model_config=args.config_file

MODEL_MAP = {
    'bert': BertForClassification,
    'bertxl': BertForClassification,
    'rnn': RnnForSentencePairClassification,
    'lr': LogisticRegression,
    'sg': FullyConnectNet,
    'cnn': CharCNN
}
if sys.hexversion < 0x03070000:
    ft = time.process_time
else:
    ft = time.process_time_ns

LABELS = []
with open('data/accusation.txt','r', encoding='utf-8') as f:
    for line in f:
        LABELS.append(line.strip())
assert(len(LABELS)==202)

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


    def bert_classification(self, content):
        logger.info('1:{}'.format(content))
        #accusation, fact
        row = {'accusation': '/', 'fact': content}
        df = pandas.DataFrame().append(row, ignore_index=True)
        filename = "data/{}.csv".format(time.time())
        df.to_csv(filename, index=False, columns=['accusation', 'fact'])
        test_set = self.data.load_file(filename, train=False)
        data_loader_test = DataLoader(
            test_set, batch_size=self.config.batch_size, shuffle=False)
        # Evaluate
        answer_list = evaluate(self.model, data_loader_test, self.device)
        answer_desc = [LABELS[i] for i in answer_list]
        return {"id": answer_list, 'desc': answer_desc}

    def on_get(self, req, resp):
        logger.info("...")
        resp.set_header('Access-Control-Allow-Origin', '*')
        resp.set_header('Access-Control-Allow-Methods', '*')
        resp.set_header('Access-Control-Allow-Headers', '*')
        resp.set_header('Access-Control-Allow-Credentials','true')
        content = req.get_param('1', True)
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
        start = ft()
        jsondata = json.loads(data)
        clean_content = jsondata['1']
        resp.media = self.bert_classification(clean_content)
        logger.info("tot:{}ns".format(ft() - start))
        logger.info("###")

if __name__=="__main__":
    api = falcon.API(middleware=[cors_allow_all.middleware])
    api.req_options.auto_parse_form_urlencoded = True
    api.add_route('/z', TorchResource())
    waitress.serve(api, port=args.port, threads=48, url_scheme='http')

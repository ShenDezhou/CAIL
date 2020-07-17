import logging
import os
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
from model import BertForClassification
from evaluate import evaluate
import time
from model import class_dic
from dataclean import cleanall, shortenlines

logging.basicConfig(level=logging.INFO, format='%(asctime)-18s %(message)s')
logger = logging.getLogger()
cors_allow_all = CORS(allow_all_origins=True,
                      allow_origins_list=['http://localhost:8081'],
                      allow_all_headers=True,
                      allow_all_methods=True,
                      allow_credentials_all_origins=True
                      )
model_config='config/bert_config.json'

class TorchResource:

    def __init__(self):
        logger.info("...")
        # 0. Load config
        with open(model_config) as fin:
            self.config = json.load(fin, object_hook=lambda d: SimpleNamespace(**d))
        if torch.cuda.is_available():
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cpu')
        # 1. Load data
        self.data = Data(vocab_file=os.path.join(self.config.model_path, 'vocab.txt'),
                    max_seq_len=self.config.max_seq_len,
                    model_type=self.config.model_type, config=self.config)

        # 2. Load model
        self.model = BertForClassification(self.config)
        self.model = load_torch_model(
            self.model, model_path=os.path.join(self.config.model_path, 'model.bin'))
        self.model.to(self.device)
        logger.info("###")


    def bert_classification(self,title, content):
        logger.info('1:{}, 2:{}'.format(title, content))
        row = {'type1': '/', 'title': title, 'content': content}
        df = pandas.DataFrame().append(row, ignore_index=True)
        filename = "data/{}.csv".format(time.time())
        df.to_csv(filename, index=False, columns=['type1', 'title', 'content'])
        test_set = self.data.load_file(filename, train=False)
        data_loader_test = DataLoader(
            test_set, batch_size=self.config.batch_size, shuffle=False)
        # Evaluate
        answer_list = evaluate(self.model, data_loader_test, self.device)
        answer_list = [class_dic[i] for i in answer_list]
        return {"answer": answer_list}

    def on_get(self, req, resp):
        logger.info("...")
        resp.set_header('Access-Control-Allow-Origin', 'http://localhost:8081')
        resp.set_header('Access-Control-Allow-Methods', '*')
        resp.set_header('Access-Control-Allow-Headers', '*')
        resp.set_header('Access-Control-Allow-Credentials','true')
        title = req.get_param('1', True)
        content = req.get_param('2', True)
        clean_title = shortenlines(title)
        clean_content = cleanall(content)
        resp.media = self.bert_classification(clean_title, clean_content)
        logger.info("###")


    def on_post(self, req, resp):
        """Handles POST requests"""
        resp.set_header('Access-Control-Allow-Origin', 'http://localhost:8081')
        resp.set_header('Access-Control-Allow-Methods', '*')
        resp.set_header('Access-Control-Allow-Headers', '*')
        resp.set_header('Access-Control-Allow-Credentials', 'true')
        resp.set_header("Cache-Control", "no-cache")
        data = req.stream.read(req.content_length)
        jsondata = json.loads(data)
        clean_title = shortenlines(jsondata.title)
        clean_content = cleanall(jsondata.content)
        resp.media = self.bert_classification(clean_title, clean_content)

if __name__=="__main__":
    api = falcon.API(middleware=[cors_allow_all.middleware])
    api.req_options.auto_parse_form_urlencoded = True
    api.add_route('/z', TorchResource())
    waitress.serve(api, port=58080, threads=48, url_scheme='http')

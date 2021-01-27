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
from tqdm import tqdm

from data import Data
from model import BertSupportNetX
from utils import load_torch_model
from tools.utils import convert_to_tokens

MODEL_MAP={
    "bert": BertSupportNetX,
    "bertxl": BertSupportNetX
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


# def result_to_json(string, tags):
#     item = {"string": string, "entities": []}
#     entity_name = ""
#     entity_start = 0
#     idx = 0
#     i = -1
#     zipped = zip(string, tags)
#     listzip = list(zipped)
#     last = len(listzip)
#     for char, tag in listzip:
#         i += 1
#         if tag == 3:
#             item["entities"].append({"word": char, "start": idx, "end": idx+1, "type":'s'})
#         elif tag == 0:
#             entity_name += char
#             entity_start = idx
#         elif tag == 1:
#             if (entity_name != "") and (i == last):
#                 entity_name += char
#                 item["entities"].append({"word": entity_name, "start": entity_start, "end": idx + 1, "type": 'bms'})
#                 entity_name = ""
#             else:
#                 entity_name += char
#         elif tag == 2:  # or i == len(zipped)
#             entity_name += char
#             item["entities"].append({"word": entity_name, "start": entity_start, "end": idx + 1, "type": 'bms'})
#             entity_name = ""
#         else:
#             entity_name = ""
#             entity_start = idx
#         idx += 1
#     return item
#


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

    def process_context(self, line):
        line = line.replace("&middot;", "", 100)
        spans = re.split('([,。])', line)
        if len(spans) <= 2:
            spans = re.split('([，。])', line)
        if len(spans) <= 2:
            spans = re.split('([;；，。,])', line)
        assert len(spans) > 2, spans
        # spans = [span for span in spans if len(span)>1]
        spans_sep = []
        for i in range(len(spans) // 2):
            spans_sep.append(spans[2 * i] + spans[2 * i + 1])
        assert len(spans_sep) > 0, spans
        return [[spans_sep[0], spans_sep]]

    def bert_classification(self, content, question):
        logger.info('1:{}'.format( content))
        conv_dic = {}
        conv_dic['_id'] = 0
        conv_dic['context'] = self.process_context(content)
        conv_dic['question'] = question
        conv_dic["answer"] = ""
        conv_dic['supporting_facts'] = []
        rows = [conv_dic]
        filename = "data/{}.json".format(time.time())
        with open(filename, 'w', encoding='utf8') as fw:
            json.dump(rows, fw, ensure_ascii=False, indent=4)

        exam, feats, dataset = self.data.load_file(filename, False)

        data_loader = DataLoader(dataset, batch_size=self.config.batch_size)

        self.model.eval()
        answer_dict = {}
        sp_dict = {}
        tqdm_obj = tqdm(data_loader, ncols=80)
        for step, batch in enumerate(tqdm_obj):
            batch = tuple(t.to(self.device) for t in batch)
            start_logits, end_logits, type_logits, sp_logits, start_position, end_position = self.model(*batch)

            batchsize = batch[0].size(0)
            # ids
            answer_dict_ = convert_to_tokens(exam, feats, batch[5], start_position.data.cpu().numpy().tolist(),
                                             end_position.data.cpu().numpy().tolist(),
                                             np.argmax(type_logits.data.cpu().numpy(), 1))
            answer_dict.update(answer_dict_)

            predict_support_np = torch.sigmoid(sp_logits).data.cpu().numpy()
            for i in range(predict_support_np.shape[0]):
                cur_sp_pred = []
                cur_id = batch[5][i].item()

                cur_sp_logit_pred = []  # for sp logit output
                for j in range(predict_support_np.shape[1]):
                    if j >= len(exam[cur_id].sent_names):
                        break

                    if predict_support_np[i, j] > self.config.sp_threshold:
                        cur_sp_pred.append(exam[cur_id].sent_names[j])
                sp_dict.update({cur_id: cur_sp_pred})

        new_answer_dict = {}
        for key, value in answer_dict.items():
            new_answer_dict[key] = value.replace(" ", "")
        prediction = {'answer': new_answer_dict, 'sp': sp_dict}

        return {"data": prediction}

    def on_get(self, req, resp):
        logger.info("...")
        resp.set_header('Access-Control-Allow-Origin', '*')
        resp.set_header('Access-Control-Allow-Methods', '*')
        resp.set_header('Access-Control-Allow-Headers', '*')
        resp.set_header('Access-Control-Allow-Credentials','true')
        content = req.get_param('c', True)
        question = req.get_param('q', True)
        # clean_content =
        #clean_content = self.cleanall(content)
        resp.media = self.bert_classification(content, question)
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
        content = jsondata['context']
        question = jsondata['question']
        # clean_content = self.cleanall(content)
        resp.media = self.bert_classification(content, question)
        logger.info("###")

if __name__=="__main__":
    api = falcon.API(middleware=[cors_allow_all.middleware])
    api.req_options.auto_parse_form_urlencoded = True
    api.add_route('/z', TorchResource())
    waitress.serve(api, port=args.port, threads=48, url_scheme='http')

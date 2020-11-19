import argparse
import logging
import re
import time

import falcon
from falcon_cors import CORS
import json
import waitress

import os
# os.chdir('sfzyy')
import sys

sys.path.append('sfzyy')
from sfzyy.main import Segment_Abstract

sys.path.remove('sfzyy')
from Sentence_Abstract import Sentence_Abstract
from Word_Abstract import Word_Abstract
from dataclean import cleanall, shortenlines

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
    '-p', '--port', default=58089,
    help='falcon server port')
parser.add_argument(
    '-c', '--config_file', default='config/bert_config.json',
    help='model config file')
args = parser.parse_args()
model_config = args.config_file


class TorchResource:

    def __init__(self):
        logger.info("...")
        self.segemnt = Segment_Abstract()
        self.sentence = Sentence_Abstract()
        self.word = Word_Abstract()
        logger.info("###")

    def process_context(self, line):
        raw_spans = line.split(r'\n')
        max_len = max([len(line) for line in raw_spans])
        if max_len > 512:
            spans_sep = []
            for line in raw_spans:
                spans = re.split('([；。])', line)
                spans = [span for span in spans if len(span) > 0]
                for i in range(len(spans) // 2):
                    spans_sep.append(spans[2 * i] + spans[2 * i + 1])
            return spans_sep
        return raw_spans

    def get_abstract(self, title, content):
        logger.info('1:{}, 2:{}'.format(title, content))
        # 0. preprocess
        sentences = self.process_context(content)
        sentences = [{"sentence": sent} for sent in sentences]
        row = {'id': title, 'text': sentences}
        # 1. segement abstract
        row = self.segemnt.get_abstract(row)
        phase1_filename = "data/{}-1.csv".format(time.time())
        with open(phase1_filename, 'w', encoding='utf8') as fw:
            fw.write(json.dumps(row, ensure_ascii=False) + '\n')

        # 2. sentence abstract
        phase2_filename = self.sentence.get_abstract(phase1_filename)

        # 3. word abstract
        summary = self.word.get_abstract(phase1_filename, phase2_filename)
        return {"answer": summary}

    def on_get(self, req, resp):
        logger.info("...")
        resp.set_header('Access-Control-Allow-Origin', '*')
        resp.set_header('Access-Control-Allow-Methods', '*')
        resp.set_header('Access-Control-Allow-Headers', '*')
        resp.set_header('Access-Control-Allow-Credentials', 'true')
        title = req.get_param('1', True)
        content = req.get_param('2', True)
        # clean_title = shortenlines(title)
        clean_content = shortenlines(content)
        resp.media = self.get_abstract(title, clean_content)
        logger.info("###")

    def on_post(self, req, resp):
        """Handles POST requests"""
        resp.set_header('Access-Control-Allow-Origin', '*')
        resp.set_header('Access-Control-Allow-Methods', '*')
        resp.set_header('Access-Control-Allow-Headers', '*')
        resp.set_header('Access-Control-Allow-Credentials', 'true')
        resp.set_header("Cache-Control", "no-cache")
        jsondata = json.loads(req.stream.read(req.content_length))
        title=jsondata['1']
        clean_content = shortenlines(jsondata['2'])
        resp.media = self.get_abstract(title, clean_content)
        logger.info("###")


if __name__ == "__main__":
    api = falcon.API(middleware=[cors_allow_all.middleware])
    api.req_options.auto_parse_form_urlencoded = True
    api.add_route('/z', TorchResource())
    waitress.serve(api, port=args.port, threads=48, url_scheme='http')

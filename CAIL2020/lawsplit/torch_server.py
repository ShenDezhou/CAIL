import argparse
import logging

import falcon

from falcon_cors import CORS
import json
import waitress

import re

import pandas
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
    '-p', '--port', default=58004,
    help='falcon server port')
# parser.add_argument(
#     '-c', '--config_file', default='config/bert_config_l.json',
#     help='model config file')
args = parser.parse_args()
# model_config=args.config_file
#
# MODEL_MAP = {
#     'bert': BertForClassification,
#     'cnn': CharCNN
# }


class TorchResource:

    def __init__(self):
        logger.info("...")

        self.rule = '　+第([^条]{1,7})条　(.*)'
        self.chapter = '第[一二三四五六七八九十]{1,3}分?[章编]'
        self.pattern = re.compile(self.rule)
        self.chapter_pattern = re.compile(self.chapter)

        self.FORMAL_DIGIT = "零一二三四五六七八九"
        self.math_digit = "0123456789"
        logger.info("###")

    def format2digit(self, word):
        trans = ""
        if word.startswith('十'):
            trans += '1'

        for c in word:
            if c in self.FORMAL_DIGIT:
                trans += self.math_digit[self.FORMAL_DIGIT.index(c)]
            if c == '千' and not word.endswith('千'):
                if '百' not in word and '十' not in word:
                    trans += "0"
            if word.endswith(c):
                if c == "十":
                    trans += '0'
                if c == "百":
                    trans += '00'
                if c == "千":
                    trans += '000'
        return trans

    def split(self, content):
        # logger.info('1:{}, 2:{}'.format(title, content))

        df = pandas.DataFrame()
        f = content.split('\n')
        buffer = []
        digit = 0
        for line in f:
            match = re.search(self.pattern, line)
            if match:
                # output
                article_digit = self.format2digit(match.group(1))
                if digit:
                    tup = (str(int(article_digit) - 1), r"\n".join(buffer))
                    buffer = []
                    dic = dict(zip(('id', 'desc'), tup))
                    df = df.append(dic, ignore_index=True)
                buffer.append(line.strip())
                digit += 1
            else:
                if self.chapter_pattern.search(line):
                    context = line.strip().split('　')[-1]
                else:
                    buffer.append(line.strip())
        # last
        if buffer:
            tup = (article_digit, r"\n".join(buffer))
            dic = dict(zip(('id', 'desc'), tup))
            df = df.append(dic, ignore_index=True)
        df.to_csv('civil_code_contract.csv', columns=['id', 'desc'], index=False)
        tuple = {'id':df['id'].to_list(), 'desc':df['desc'].to_list()}
        return tuple


    def on_get(self, req, resp):
        logger.info("...")
        resp.set_header('Access-Control-Allow-Origin', '*')
        resp.set_header('Access-Control-Allow-Methods', '*')
        resp.set_header('Access-Control-Allow-Headers', '*')
        resp.set_header('Access-Control-Allow-Credentials','true')
        # title = req.get_param('1', True)
        content = req.get_param('1', True)
        # clean_title = shortenlines(title)
        # clean_content = cleanall(content)
        resp.media = self.split(content)
        logger.info("###")


    def on_post(self, req, resp):
        """Handles POST requests"""
        resp.set_header('Access-Control-Allow-Origin', '*')
        resp.set_header('Access-Control-Allow-Methods', '*')
        resp.set_header('Access-Control-Allow-Headers', '*')
        resp.set_header('Access-Control-Allow-Credentials', 'true')
        resp.set_header("Cache-Control", "no-cache")
        data = req.stream.read(req.content_length)
        jsondata = json.loads(data)
        # clean_title = shortenlines(jsondata['title'])
        # clean_content = self.split((jsondata['content'])
        resp.media = self.split(jsondata['content'])

if __name__=="__main__":
    api = falcon.API(middleware=[cors_allow_all.middleware])
    api.req_options.auto_parse_form_urlencoded = True
    api.add_route('/z', TorchResource())
    waitress.serve(api, port=args.port, threads=48, url_scheme='http')

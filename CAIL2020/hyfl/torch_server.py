import argparse
import logging
import os
import sys
import time
from types import SimpleNamespace
import falcon
import lawrouge
import pandas
from falcon_cors import CORS
import json
import waitress


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
    '-p', '--port', default=58071,
    help='falcon server port')
args = parser.parse_args()
if sys.hexversion < 0x03070000:
    ft = time.process_time
else:
    ft = time.process_time_ns


class TorchResource:

    def __init__(self):
        logger.info("...")

        self.rouge = lawrouge.Rouge(isChinese=True)
        self.df = pandas.read_csv('data/category.csv')
        self.supreme = pandas.read_csv('data/supreme_dic.csv')
        self.supreme['start'] = self.supreme['start'].astype(int)
        self.supreme['end'] = self.supreme['end'].astype(int)
        logger.info("###")

    def get_Supreme(self, subclass):
        subclass = int(subclass)
        sup = self.supreme[(self.supreme['start'] <= subclass) & (self.supreme['end']>= subclass)]
        if len(sup):
            sup = sup.iloc[0]
            return sup.to_dict()
        return {'id':'',"desc":''}

    def rouge_classification(self, title):
        def get_score(candidates):
            line = title
            if candidates:
                candidates = str(candidates).split(r'\n')
                if len(candidates) > 1:
                    scores = [self.rouge.get_scores([line], [candidate], avg=2) for candidate in candidates]
                    scores = [score['f'] for score in scores]
                    return max(scores)
                else:
                    score = self.rouge.get_scores([line], [str(candidates)], avg=2)
                    return score['f']
            return 0

        self.df['score'] = self.df['desc'].map(get_score).astype(float)
        index = self.df['score'].argmax()
        subinfo = self.df.iloc[index].to_dict()
        if len(subinfo['id'])>2:
            supinfo = self.get_Supreme(subinfo['id'][:2])
            subinfo['sup_id'] = supinfo['id']
            subinfo['sup_desc'] = supinfo['desc']
        else:
            subinfo['sup_id'] = subinfo['id']
            subinfo['sup_desc'] = subinfo['desc']
        return subinfo


    def on_get(self, req, resp):
        logger.info("...")
        resp.set_header('Access-Control-Allow-Origin', '*')
        resp.set_header('Access-Control-Allow-Methods', '*')
        resp.set_header('Access-Control-Allow-Headers', '*')
        resp.set_header('Access-Control-Allow-Credentials','true')
        title = req.get_param('1', True)
        start = ft()
        resp.media = self.rouge_classification(title)
        logger.info("tot:{}ns".format(ft() - start))
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
        start = ft()
        clean_title = jsondata['1']
        resp.media = self.rouge_classification(clean_title)
        logger.info("tot:{}ns".format(ft() - start))
        logger.info("###")

if __name__=="__main__":
    api = falcon.API(middleware=[cors_allow_all.middleware])
    api.req_options.auto_parse_form_urlencoded = True
    api.add_route('/z', TorchResource())
    waitress.serve(api, port=args.port, threads=48, url_scheme='http')

import argparse
import logging

import falcon

from falcon_cors import CORS
import json
import waitress
from articlebyarticle import TypeArticle

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
    '-p', '--port', default=58098,
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
        self.typearticle = TypeArticle()
        logger.info("###")

    def getIntention(self, contract_title, contract_article):
        logger.info("...")
        index, typename = self.typearticle.get_type(contract_title)
        print(index, typename)
        #contract_article = '3、违反本合同第四条第4款，延期交付的违约方应按延期交付天数每天人民币_________元向乙方支付违约金，并承担延误期内的风险责任。'
        intention_articles = self.typearticle.findArticles(typename)
        print(intention_articles)
        index, article = self.typearticle.get_weighted_score(contract_article, intention_articles)
        print(index, article)

        default_intention_articles = self.typearticle.findArticles_default_2(contract_article)
        print(default_intention_articles)
        index, article_default = self.typearticle.get_weighted_score(contract_article, default_intention_articles)
        print(index, article)
        logger.info("###")
        return {"special":article,"special_all":intention_articles, "common": article_default, "common_all": default_intention_articles}


    def on_get(self, req, resp):
        logger.info("...")
        resp.set_header('Access-Control-Allow-Origin', '*')
        resp.set_header('Access-Control-Allow-Methods', '*')
        resp.set_header('Access-Control-Allow-Headers', '*')
        resp.set_header('Access-Control-Allow-Credentials','true')
        title = req.get_param('1', True)
        content = req.get_param('2', True)
        # clean_title = shortenlines(title)
        # clean_content = cleanall(content)
        resp.media = self.getIntention(title, content)
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
        resp.media = self.getIntention(jsondata['title'], jsondata['article'])

if __name__=="__main__":
    api = falcon.API(middleware=[cors_allow_all.middleware])
    api.req_options.auto_parse_form_urlencoded = True
    api.add_route('/z', TorchResource())
    waitress.serve(api, port=args.port, threads=48, url_scheme='http')

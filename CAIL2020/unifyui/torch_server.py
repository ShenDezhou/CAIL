import argparse
import logging

import falcon

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
    '-p', '--port', default=58100,
    help='falcon server port')

args = parser.parse_args()


class RootResource:

    def __init__(self):
        logger.info("...")

    def on_get(self, req, resp, filename="default.html"):
        # do some sanity check on the filename

        resp.status = falcon.HTTP_200
        resp.content_type = 'text/html'
        with open(filename, 'r', encoding='utf-8') as f:
            resp.body = f.read()

class TorchResource:

    def __init__(self):
        logger.info("...")


    def on_get(self, req, resp, filename):
        # do some sanity check on the filename

        resp.status = falcon.HTTP_200
        resp.content_type = 'text/html'
        with open(filename, 'r', encoding='utf-8') as f:
            resp.body = f.read()

if __name__=="__main__":
    api = falcon.API(middleware=[cors_allow_all.middleware])
    api.req_options.auto_parse_form_urlencoded = True
    api.add_route('/', RootResource())
    api.add_route('/static/{filename}', TorchResource())
    waitress.serve(api, port=args.port, threads=48, url_scheme='http')

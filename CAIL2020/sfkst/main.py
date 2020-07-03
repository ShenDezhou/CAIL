import argparse
import os
import logging
import json

from init_tool import init_all
from config_parser import create_config
from test_tool import test

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', default='default.config', help="specific config file", required=False)
    parser.add_argument('--gpu', '-g', default="0", help="gpu id list")
    parser.add_argument('--checkpoint', default='model/bert/model0.bin', help="checkpoint file path", required=False)
    parser.add_argument('--input', default='/input/', help="input file path", required=False)
    parser.add_argument('--result', default='/output/result.txt', help="result file path", required=False)
    args = parser.parse_args()

    config = create_config(args.config)

    config.set("data", "test_data_path", args.input)
    fs = ",".join(os.listdir(args.input))
    config.set("data", "test_file_list", fs)

    gpu_list = []
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        device_list = args.gpu.split(",")
        for a in range(0, len(device_list)):
            gpu_list.append(int(a))

    parameters = init_all(config, gpu_list, args.checkpoint, "test")
    predict = test(parameters, config, gpu_list)
    json.dump(predict, open(args.result, "w", encoding="utf8"), ensure_ascii=False,
              sort_keys=True, indent=2)

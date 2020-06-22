import json
import os
import random


input_path = "input"
output_path = "output/demo-result.txt"

if __name__ == "__main__":
    
    result = {}
    for filename in os.listdir(input_path):
        data = []
        for line in open(os.path.join(input_path, filename), "r", encoding="utf8"):
            data.append(json.loads(line))
        for item in data:
            id = item["id"]
            result[id] = []
            for option in ["A", "B", "C", "D"]:
                if random.randint(1, 2) == 1:
                    result[id].append(option)

    json.dump(result, open(output_path, "w", encoding="utf8"), indent=2, ensure_ascii=False, sort_keys=True)

import argparse
import os
import torch
import logging
import json

from tools.init_tool import init_all
from config_parser import create_config
from tools.test_tool import test

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', default='default.config', help="specific config file", required=False)
    parser.add_argument('--gpu', '-g', help="gpu id list")
    parser.add_argument('--checkpoint', default='saved_model/bert/model.bin', help="checkpoint file path", required=False)
    parser.add_argument('--result', default='output/result.txt', help="result file path", required=False)
    args = parser.parse_args()

    configFilePath = args.config


    use_gpu = True
    gpu_list = []
    if args.gpu is None:
        use_gpu = False
    else:
        use_gpu = True
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

        device_list = args.gpu.split(",")
        for a in range(0, len(device_list)):
            gpu_list.append(int(a))

    os.system("clear")

    config = create_config(configFilePath)
    input_path = "/input"
    input_files = [os.path.join(input_path, filename) for filename in os.listdir(input_path)]


    cuda = torch.cuda.is_available()
    logger.info("CUDA available: %s" % str(cuda))
    if not cuda and len(gpu_list) > 0:
        logger.error("CUDA is not available but specific gpu id")
        raise NotImplementedError

    parameters = init_all(config, gpu_list, args.checkpoint, "test")
    # parameters = [init_all(config, gpu_list, args.checkpoint.replace('.pkl','s.pkl'), "test"), init_all(config, gpu_list, args.checkpoint.replace('.pkl','m.pkl'), "test")]
    json.dump(test(parameters, config, gpu_list), open(args.result, "w", encoding="utf8"), ensure_ascii=False,
              sort_keys=True, indent=2)

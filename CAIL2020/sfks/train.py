import argparse
import os
import torch
import logging

from tools.init_tool import init_all
from config_parser import create_config
from tools.train_tool import train

# from gbt.SingleMulti import SingleMulti
from torch.autograd import Variable

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', default="default.config", help="specific config file", required=False)
    parser.add_argument('--gpu', '-g', help="gpu id list")
    parser.add_argument('--checkpoint', help="checkpoint file path")
    parser.add_argument('--do_test', help="do test while training or not", default="False", action="store_true")
    args = parser.parse_args()

    gpu_list = []
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        device_list = args.gpu.split(",")
        for a in range(0, len(device_list)):
            gpu_list.append(int(a))

    config = create_config(args.config)

    if not torch.cuda.is_available() and len(gpu_list) > 0:
        logger.error("CUDA is not available but specific gpu id")
        raise NotImplementedError
    else:
        logger.info("CUDA available")

    parameters = init_all(config, gpu_list, args.checkpoint, "train")
    train(parameters, config, gpu_list, args.do_test)

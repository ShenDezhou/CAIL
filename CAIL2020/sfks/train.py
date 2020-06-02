import argparse
import os
import torch
import logging

from tools.init_tool import init_all
from config_parser import create_config
from tools.train_tool import train

from gbt.SingleMulti import SingleMulti

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', default="default.config", help="specific config file", required=False)
    parser.add_argument('--gpu', '-g', help="gpu id list")
    parser.add_argument('--checkpoint', help="checkpoint file path")
    parser.add_argument('--do_test', help="do test while training or not", action="store_true")
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


    os.system("cls")

    config = create_config(configFilePath)

    cuda = torch.cuda.is_available()
    logger.info("CUDA available: %s" % str(cuda))
    if not cuda and len(gpu_list) > 0:
        logger.error("CUDA is not available but specific gpu id")
        raise NotImplementedError


    parameters = init_all(config, gpu_list, args.checkpoint, "train")

    do_test = False
    if args.do_test:
        do_test = True

    train(parameters, config, gpu_list, do_test)

import argparse
import os
import sys
import pathlib

import torch
from torch import nn
from torchvision import transforms
from torchocr.networks import build_model
from torchocr.datasets.det_modules import ResizeShortSize
from torchocr.postprocess import build_post_process
import cv2
from matplotlib import pyplot as plt
from torchocr.utils import draw_ocr_box_txt, draw_bbox


class DetInfer:
    def __init__(self, model_path):
        ckpt = torch.load(model_path, map_location='cpu')
        cfg = ckpt['cfg']
        self.model = build_model(cfg.model)
        state_dict = {}
        for k, v in ckpt['state_dict'].items():
            state_dict[k.replace('module.', '')] = v
        self.model.load_state_dict(state_dict)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

        self.resize = ResizeShortSize(736, False)
        self.post_proess = build_post_process(cfg.post_process)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.dataset.train.dataset.mean, std=cfg.dataset.train.dataset.std)
        ])

    def predict(self, img, is_output_polygon=False):
        # 预处理根据训练来
        data = {'img': img, 'shape': [img.shape[:2]], 'text_polys': []}
        data = self.resize(data)
        tensor = self.transform(data['img'])
        tensor = tensor.unsqueeze(dim=0)
        tensor = tensor.to(self.device)
        out = self.model(tensor)
        box_list, score_list = self.post_proess(out, data['shape'], is_output_polygon=is_output_polygon)
        box_list, score_list = box_list[0], score_list[0]
        if len(box_list) > 0:
            idx = [x.sum() > 0 for x in box_list]
            box_list = [box_list[i] for i, v in enumerate(idx) if v]
            score_list = [score_list[i] for i, v in enumerate(idx) if v]
        else:
            box_list, score_list = [], []
        return box_list, score_list


def init_args():
    import argparse
    parser = argparse.ArgumentParser(description='PytorchOCR infer')
    parser.add_argument('--model_path', required=False, type=str, help='rec model path', default=r'F:\CAIL\CAIL2020\cocr\model\db_ResNet50_vd_icdar2015withconfig.pth')
    parser.add_argument('--img_path', required=False, type=str, help='img path for predict', default=r'F:\CAIL\CAIL2020\cocr\data\icdar2015\detection\test\imgs\img_2.jpg')
    args = parser.parse_args()
    return args

def resize(img, scale_percent = 60):
    scale_percent = 60  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized


if __name__ == '__main__':
    # ===> 获取配置文件参数
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--config', type=str, default='config/det.json',
                        help='train config file path')
    parser.add_argument('-m','--model_path', required=False, type=str, help='rec model path', default=r'F:\CAIL\CAIL2020\cocr\model\det-model.bin')
    parser.add_argument('-i','--img_path', required=False, type=str, help='img path for predict', default=r'F:\CAIL\CAIL2020\cocr\data\t2\architecture (1).jpg')
    args = parser.parse_args()

    # for i in range(1,11):
    img = cv2.imread(args.img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.shape[0] > 1500:
        img = resize(img, img.shape[0]*100./1024)
    model = DetInfer(args.model_path)
    box_list, score_list = model.predict(img, is_output_polygon=True)
    img = draw_ocr_box_txt(img, box_list)
    img = draw_bbox(img, box_list)
    plt.imshow(img)
    plt.show()

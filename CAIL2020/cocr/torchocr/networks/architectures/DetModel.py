# -*- coding: utf-8 -*-
# @Time    : 2020/5/21 14:23
# @Author  : THU
from addict import Dict as AttrDict
from torch import nn

from torchocr.networks.backbones.DetMobilenetV3 import MobileNetV3
from torchocr.networks.backbones.DetResNetvd import ResNet
from torchocr.networks.necks.FPN import FPN
from torchocr.networks.heads.DetDbHead import DBHead

backbone_dict = {'MobileNetV3': MobileNetV3, 'ResNet': ResNet}
neck_dict = {'FPN': FPN}
head_dict = {'DBHead': DBHead}


class DetModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.in_channels, 'in_channels must in model config'
        backbone_type = config.backbone.type#.pop('type')
        assert backbone_type in backbone_dict, f'backbone.type must in {backbone_dict}'
        self.backbone = backbone_dict[backbone_type](config.in_channels, **config.backbone.__dict__)

        neck_type = config.neck.type#.pop('type')
        assert neck_type in neck_dict, f'neck.type must in {neck_dict}'
        self.neck = neck_dict[neck_type](self.backbone.out_channels, **config.neck.__dict__)

        head_type = config.head.type#.pop('type')
        assert head_type in head_dict, f'head.type must in {head_dict}'
        self.head = head_dict[head_type](self.neck.out_channels, **config.head.__dict__)

        self.name = f'DetModel_{backbone_type}_{neck_type}_{head_type}'

    def load_3rd_state_dict(self, _3rd_name, _state):
        self.backbone.load_3rd_state_dict(_3rd_name, _state)
        self.neck.load_3rd_state_dict(_3rd_name, _state)
        self.head.load_3rd_state_dict(_3rd_name, _state)

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x


if __name__ == '__main__':
    import torch

    db_config = AttrDict(
        in_channels=3,
        backbone=AttrDict(type='MobileNetV3', layers=50, model_name='large',pretrained=True),
        neck=AttrDict(type='FPN', out_channels=256),
        head=AttrDict(type='DBHead')
    )
    x = torch.zeros(1, 3, 640, 640)
    model = DetModel(db_config)

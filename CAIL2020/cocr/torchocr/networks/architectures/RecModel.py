# -*- coding: utf-8 -*-
# @Time    : 2020/5/16 11:18
# @Author  : THU
from torch import nn

from torchocr.networks.backbones.RecMobileNetV3 import MobileNetV3
from torchocr.networks.backbones.RecResNetvd import ResNet
from torchocr.networks.backbones.RecResNetLMvd import ResBertNet
from torchocr.networks.necks.RecSequenceDecoder import SequenceDecoder, Reshape
from torchocr.networks.heads.RecCTCHead import CTC

backbone_dict = {'MobileNetV3': MobileNetV3, 'ResNet': ResNet, 'ResBertNet': ResBertNet}
neck_dict = {'RNN': SequenceDecoder, 'None': Reshape}
head_dict = {'CTC': CTC}


class RecModel(nn.Module):
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

        self.name = f'RecModel_{backbone_type}_{neck_type}_{head_type}'

    def load_3rd_state_dict(self, _3rd_name, _state):
        self.backbone.load_3rd_state_dict(_3rd_name, _state)
        self.neck.load_3rd_state_dict(_3rd_name, _state)
        self.head.load_3rd_state_dict(_3rd_name, _state)

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x
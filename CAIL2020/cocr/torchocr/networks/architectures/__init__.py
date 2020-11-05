# -*- coding: utf-8 -*-
# @Time    : 2020/5/15 17:42
# @Author  : THU
from addict import Dict
import copy
from .RecModel import RecModel
from .DetModel import DetModel

support_model = ['RecModel', 'DetModel']


def build_model(config):
    """
    get architecture model class
    """
    copy_config = copy.deepcopy(config)
    arch_type = copy_config.type #.pop("type"_
    assert arch_type in support_model, f'{arch_type} is not developed yet!, only {support_model} are support now'
    arch_model = eval(arch_type)(config)
    return arch_model

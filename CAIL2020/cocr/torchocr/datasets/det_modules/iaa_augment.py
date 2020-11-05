# -*- coding: utf-8 -*-
# @Time    : 2019/12/4 18:06
# @Author  : THU
import numpy as np
import imgaug
import imgaug.augmenters as iaa


class AugmenterBuilder(object):
    def __init__(self):
        pass

    def build(self, args, root=True):
        if args is None:
            return None
        elif isinstance(args, list):
            if len(args) == 0:
                return None
            if root:
                sequence = [self.build(value.__dict__, root=False) for value in args]
                return iaa.Sequential(sequence)
            else:
                return getattr(iaa, args[0])(*[self.to_tuple_if_list(a) for a in args[1:]])
        elif isinstance(args, dict):
            cls = getattr(iaa, args['type'])
            return cls(**{k: self.to_tuple_if_list(v) for k, v in args['args'].__dict__.items()})
        else:
            raise RuntimeError('unknown augmenter arg: ' + str(args))

    def to_tuple_if_list(self, obj):
        if isinstance(obj, list):
            return tuple(obj)
        return obj


class IaaAugment():
    def __init__(self, augmenter_args=None):
        if augmenter_args is None:
            augmenter_args = [{'type': 'Fliplr', 'args': {'p': 0.5}},
                              {'type': 'Affine', 'args': {'rotate': [-10, 10]}},
                              {'type': 'Resize', 'args': {'size': [0.5, 3]}}]
        self.augmenter = AugmenterBuilder().build(augmenter_args)

    def __call__(self, data):
        image = data['img']
        shape = image.shape

        if self.augmenter:
            aug = self.augmenter.to_deterministic()
            data['img'] = aug.augment_image(image)
            data = self.may_augment_annotation(aug, data, shape)
        return data

    def may_augment_annotation(self, aug, data, shape):
        if aug is None:
            return data

        line_polys = []
        for poly in data['text_polys']:
            new_poly = self.may_augment_poly(aug, shape, poly)
            line_polys.append(new_poly)
        data['text_polys'] = np.array(line_polys)
        return data

    def may_augment_poly(self, aug, img_shape, poly):
        keypoints = [imgaug.Keypoint(p[0], p[1]) for p in poly]
        keypoints = aug.augment_keypoints(
            [imgaug.KeypointsOnImage(keypoints, shape=img_shape)])[0].keypoints
        poly = [(p.x, p.y) for p in keypoints]
        return poly
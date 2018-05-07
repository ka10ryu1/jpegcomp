#!/usr/bin/env python3
# -*-coding: utf-8 -*-
#
help = 'モデルの枝刈をする'
#

import numpy

import chainer
import chainer.links as L
from chainer import training
import chainer.cuda


def create_layer_mask(weights, pruning_rate, xp=chainer.cuda.cupy):

    if weights.data is None:
        raise Exception("Some weights of layer is None.")

    abs_W = xp.abs(weights.data)
    data = xp.sort(xp.ndarray.flatten(abs_W))
    num_prune = int(len(data) * pruning_rate)
    idx_prune = min(num_prune, len(data)-1)
    threshould = data[idx_prune]

    mask = abs_W
    mask[mask < threshould] = 0
    mask[mask >= threshould] = 1
    return mask


'''Returns a trainer extension to fix pruned weight of the model.
'''


def create_model_mask(model, pruning_rate, gpu_id):
    masks = {}
    xp = numpy
    if gpu_id >= 0:
        xp = chainer.cuda.cupy

    for name, link in model.namedlinks():
        # specify pruned layer
        if type(link) not in (L.Convolution2D, L.Linear):
            continue
        mask = create_layer_mask(link.W, pruning_rate, xp)
        masks[name] = mask
    return masks


def prune_weight(model, masks):
    for name, link in model.namedlinks():
        if name not in masks.keys():
            continue
        mask = masks[name]
        link.W.data = link.W.data * mask


'''Returns a trainer extension to fix pruned weight of the model.
'''


def pruned(model, masks):
    @training.make_extension(trigger=(1, 'iteration'))
    def _pruned(trainer):
        prune_weight(model, masks)
    return _pruned

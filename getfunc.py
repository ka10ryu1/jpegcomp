#!/usr/bin/env python3
# -*-coding: utf-8 -*-
#
help = '画像処理に関する便利機能'
#

import os
import sys
import json

import chainer.functions as F
import chainer.optimizers as O

[sys.path.append(d) for d in ['./Tools/', '../Tools/'] if os.path.isdir(d)]
from func import fileFuncLine


def lossfun(lossfun_str):
    """
    入力文字列から損失関数を推定する
    """

    if(lossfun_str.lower() == 'mse'):
        lossfun = F.mean_squared_error
    elif(lossfun_str.lower() == 'mae'):
        lossfun = F.mean_absolute_error
    elif(lossfun_str.lower() == 'ber'):
        lossfun = F.bernoulli_nll
    elif(lossfun_str.lower() == 'gauss_kl'):
        lossfun = F.gaussian_kl_divergence
    else:
        lossfun = F.mean_squared_error
        print('\n[Warning] {0}\n\t{1}->{2}\n'.format(
            fileFuncLine(), lossfun_str, lossfun.__name__)
        )

    print('Loss func:', lossfun.__name__)
    return lossfun


def F_None(x):
    return x


def actfun(actfun_str):
    """
    入力文字列から活性化関数を推定する
    """

    if(actfun_str.lower() == 'relu'):
        actfun = F.relu
    elif(actfun_str.lower() == 'elu'):
        actfun = F.elu
    elif(actfun_str.lower() == 'c_relu'):
        actfun = F.clipped_relu
    elif(actfun_str.lower() == 'l_relu'):
        actfun = F.leaky_relu
    elif(actfun_str.lower() == 'sigmoid'):
        actfun = F.sigmoid
    elif(actfun_str.lower() == 'h_sigmoid'):
        actfun = F.hard_sigmoid
    elif(actfun_str.lower() == 'tanh'):
        actfun = F.tanh
    elif(actfun_str.lower() == 's_plus'):
        actfun = F.softplus
    elif(actfun_str.lower() == 'none'):
        actfun = F_None
    else:
        actfun = F.relu
        print('\n[Warning] {0}\n\t{1}->{2}\n'.format(
            fileFuncLine(), actfun_str, actfun.__name__)
        )

    print('Activation func:', actfun.__name__)
    return actfun


def optimizer(opt_str):
    """
    入力文字列からオプティマイザを推定する
    """

    if(opt_str.lower() == 'adam'):
        opt = O.Adam()
    elif(opt_str.lower() == 'ada_d'):
        opt = O.AdaDelta()
    elif(opt_str.lower() == 'ada_g'):
        opt = O.AdaGrad()
    elif(opt_str.lower() == 'm_sgd'):
        opt = O.MomentumSGD()
    elif(opt_str.lower() == 'n_ag'):
        opt = O.NesterovAG()
    elif(opt_str.lower() == 'rmsp'):
        opt = O.RMSprop()
    elif(opt_str.lower() == 'rmsp_g'):
        opt = O.RMSpropGraves()
    elif(opt_str.lower() == 'sgd'):
        opt = O.SGD()
    elif(opt_str.lower() == 'smorms'):
        opt = O.SMORMS3()
    else:
        opt = O.Adam()
        print('\n[Warning] {0}\n\t{1}->{2}\n'.format(
            fileFuncLine(), opt_str, opt.__doc__.split('.')[0])
        )

    print('Optimizer:', opt.__doc__.split('.')[0])
    return opt


def modelParam(path):
    """
    jsonで記述されたモデルパラメータ情報を読み込む
    [in]  path:              jsonファイルのパス
    [out] d['network']:      ネットワークの種類
    [out] d['unut']:         中間層のユニット数
    [out] ch:                画像のチャンネル数
    [out] size:              画像の分割サイズ
    [out] d['layer_num']:    ネットワーク層の数
    [out] d['shuffle_rate']: PSのshuffle rate
    [out] af1:               活性化関数(1)
    [out] af2:               活性化関数(2)
    """

    print('model param:', path)
    try:
        with open(path, 'r') as f:
            d = json.load(f)

    except:
        import traceback
        traceback.print_exc()
        print(fileFuncLine())
        exit()

    if 'network' in d:
        net = d['network']
    else:
        net = 'None'

    if 'layer_num' in d:
        layer = d['layer_num']
    else:
        layer = 0

    af1 = actfun(d['actfun1'])
    af2 = actfun(d['actfun2'])
    ch = d['shape'][0]
    size = d['shape'][1]
    return \
        net, d['unit'], ch, size, \
        layer, d['shuffle_rate'], af1, af2

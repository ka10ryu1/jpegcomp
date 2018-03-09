#!/usr/bin/env python3
# -*-coding: utf-8 -*-
#
help = '画像処理に関する便利機能'
#

import os
import sys
import cv2
import json
import numpy as np

try:
    import cupy as xp
except ImportError:
    print('not import cupy')

import chainer.functions as F
import chainer.optimizers as O

[sys.path.append(d) for d in ['./Tools/', '../Tools/'] if os.path.isdir(d)]
from func import fileFuncLine


def getCh(ch):
    """
    入力されたチャンネル数をOpenCVの形式に変換する
    [in]  ch:入力されたチャンネル数 (type=int or np.shape)
    [out] OpenCVの形式
    """

    if(ch == 1):
        return cv2.IMREAD_GRAYSCALE
    elif(ch == 3):
        return cv2.IMREAD_COLOR
    else:
        return cv2.IMREAD_UNCHANGED


def isImage(name):
    """
    入力されたパスが画像か判定する
    [in]  name: 画像か判定したいパス
    [out] 画像ならTrue
    """

    # cv2.imreadしてNoneが返ってきたら画像でないとする
    if cv2.imread(name) is not None:
        return True
    else:
        print('[{0}] is not Image'.format(name))
        print(fileFuncLine())
        return False


def encodeDecode(in_imgs, ch, quality=5):
    """
    入力された画像リストを圧縮する
    [in]  in_imgs:  入力画像リスト
    [in]  ch:       出力画像リストのチャンネル数 （OpenCV形式）
    [in]  quality:  圧縮する品質 (1-100)
    [out] out_imgs: 出力画像リスト
    """

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    out_imgs = []

    for img in in_imgs:
        result, encimg = cv2.imencode('.jpg', img, encode_param)
        if False == result:
            print('[Error] {0}\n\tcould not encode image!'.format(
                fileFuncLine())
            )
            exit()

        decimg = cv2.imdecode(encimg, ch)
        out_imgs.append(decimg)

    return out_imgs


def split(imgs, size, round_num=-1, flg=cv2.BORDER_REPLICATE):
    """
    入力された画像リストを正方形に分割する
    imgsに格納されている画像はサイズが同じであること
    [in]  imgs:      入力画像リスト
    [in]  size:      正方形のサイズ（size x size）
    [in]  round_num: 丸める画像数
    [in]  flg:       境界線のフラグ
    [out] 分割されたnp.array形式の正方形画像リスト
    """

    # 画像を分割する際に端が切れてしまうのを防ぐために余白を追加する
    imgs = [cv2.copyMakeBorder(img, 0, size, 0, size, flg)
            for img in imgs]
    # 画像を分割しやすいように画像サイズを変更する
    v_size = [img.shape[0] // size * size for img in imgs]
    h_size = [img.shape[1] // size * size for img in imgs]
    imgs = [i[:v, :h] for i, v, h in zip(imgs, v_size, h_size)]
    # 画像の分割数を計算する
    v_split = [img.shape[0] // size for img in imgs]
    h_split = [img.shape[1] // size for img in imgs]
    # 画像を分割する
    out_imgs = []
    [[out_imgs.extend(np.vsplit(hi, v))
      for hi in np.hsplit(i, h)]
     for i, h, v in zip(imgs, h_split, v_split)]

    # 切り捨てたい数よりも画像数が少ないと0枚になってしまうので注意
    if(round_num > len(out_imgs)):
        print('[Error] round({0}) > split images({1})'.format(
            round_num, len(out_imgs)))
        print(fileFuncLine())
        exit()

    # バッチサイズの関係などで、画像の数を調整したい時はここで調整する
    # predict.pyなどで分割画像を復元したくなるので縦横の分割数も返す
    if(round_num > 0):
        round_len = len(out_imgs) // round_num * round_num
        return np.array(out_imgs[:round_len]), (v_split[0], h_split[0])
    else:
        return np.array(out_imgs), (v_split[0], h_split[0])


def rotate(imgs, num=2):
    """
    画像を回転させてデータ数を水増しする
    [in]  imgs:     入力画像リスト
    [in]  num:      水増しする数（最大4倍）
    [out] out_imgs: 出力画像リスト
    """

    # ベース
    out_imgs = imgs.copy()
    # 上下反転を追加
    [out_imgs.append(cv2.flip(i, 0)) for i in imgs]
    if(num > 1):
        # 左右反転を追加
        [out_imgs.append(cv2.flip(i, 1)) for i in imgs]

    if(num > 2):
        # 上下左右反転を追加
        [out_imgs.append(cv2.flip(cv2.flip(i, 1), 0)) for i in imgs]

    return out_imgs


def whiteCheck(imgs, val=245):
    """
    画像リストのうち、ほとんど白い画像を除去する
    [in] imgs: 判定する画像リスト
    [in] val:  除去するしきい値
    [out] ほとんど白い画像を除去した画像リスト
    """

    return [i for i in imgs
            if(val > np.sum(i) // (i.shape[0] * i.shape[1]))]


def resize(img, rate, flg=cv2.INTER_NEAREST):
    """
    画像サイズを変更する
    [in] img:  N倍にする画像
    [in] rate: 倍率
    [in] flg:  N倍にする時のフラグ
    [out] N倍にされた画像リスト
    """

    size = (int(img.shape[1] * rate),
            int(img.shape[0] * rate))
    return cv2.resize(img, size, flg)


def size2x(imgs, flg=cv2.INTER_NEAREST):
    """
    画像のサイズを2倍にする
    [in] imgs: 2倍にする画像リスト
    [in] flg:  2倍にする時のフラグ
    [out] 2倍にされた画像リスト
    """

    return [resize(i, 2, flg) for i in imgs]


def arr2x(arr, flg=cv2.INTER_NEAREST):
    """
    行列を画像に変換し、サイズを2倍にする
    [in] arr: 2倍にする行列
    [in] flg: 2倍にする時のフラグ
    [out] 2倍にされた行列
    """

    imgs = arr2imgs(arr)
    return imgs2arr(size2x(imgs))


def imgs2arr(imgs, norm=255, dtype=np.float32, gpu=-1):
    """
    入力画像リストをChainerで利用するために変換する
    [in]  imgs:  入力画像リスト
    [in]  norm:  正規化する値（255であれば、0-255を0-1に正規化する）
    [in]  dtype: 変換するデータタイプ
    [in]  gpu:   GPUを使用する場合はGPUIDを入力する
    [out] 生成された行列
    """

    try:
        w, h, ch = imgs[0].shape
    except:
        w, h = imgs[0].shape
        ch = 1

    if(gpu >= 0):
        return xp.array(imgs, dtype=dtype).reshape((-1, ch, w, h)) / norm
    else:
        return np.array(imgs, dtype=dtype).reshape((-1, ch, w, h)) / norm


def arr2imgs(arr, norm=255, dtype=np.uint8):
    """
    Chainerの出力をOpenCVで可視化するために変換する
    [in]  arr:   Chainerから出力された行列
    [in]  norm:  正規化をもとに戻す数（255であれば、0-1を0-255に変換する）
    [in]  dtype: 変換するデータタイプ
    [out] OpenCV形式の画像に変換された行列
    """

    ch, size = arr.shape[1], arr.shape[2]
    y = np.array(arr).reshape((-1, size, size, ch)) * 255
    return np.array(y, dtype=np.uint8)


def getLossfun(lossfun_str):
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


def getActfun(actfun_str):
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


def getOptimizer(opt_str):
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


def getModelParam(path):
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

    af1 = getActfun(d['actfun_1'])
    af2 = getActfun(d['actfun_2'])
    ch = d['shape'][0]
    size = d['shape'][1]
    return \
        d['network'], d['unit'], ch, size, \
        d['layer_num'], d['shuffle_rate'], af1, af2

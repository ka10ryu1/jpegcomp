#!/usr/bin/env python3
help = '便利機能'

import os
import cv2
import numpy as np

try:
    import cupy as xp
except ImportError:
    print('not import cupy')

import chainer.functions as F


def argsPrint(p, bar=30):
    """
    argparseの parse_args() で生成されたオブジェクトを入力すると、
    integersとaccumulateを自動で取得して表示する
    [in] p: parse_args()で生成されたオブジェクト
    [in] bar: 区切りのハイフンの数
    """

    print('-' * bar)
    args = [(i, getattr(p, i)) for i in dir(p) if not '_' in i[0]]
    for i, j in args:
        if isinstance(j, list):
            print('{0}[{1}]:'.format(i, len(j)))
            [print('\t{}'.format(k)) for k in j]
        else:
            print('{0}:\t{1}'.format(i, j))

    print('-' * bar)


def getCh(ch):
    """
    入力されたチャンネル数をOpenCVの形式に変換する
    [in]  ch:入力されたチャンネル数 (type=int)
    [out] OpenCVの形式
    """

    if(ch == 1):
        return cv2.IMREAD_GRAYSCALE
    elif(ch == 3):
        return cv2.IMREAD_COLOR
    else:
        return cv2.IMREAD_UNCHANGED


def imgEncodeDecode(in_imgs, ch, quality=5):
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
            print('could not encode image!')
            exit()

        decimg = cv2.imdecode(encimg, ch)
        out_imgs.append(decimg)

    return out_imgs


def imgSplit(imgs, size, round_num=-1, flg=cv2.BORDER_REFLECT_101):
    """
    入力された画像リストを正方形に分割する
    imgsに格納されている画像はサイズが同じであること
    [in]  imgs:      入力画像リスト
    [in]  size:      正方形のサイズ（size x size）
    [in]  round_num: 丸める画像数
    [out] 分割されたnp.array形式の正方形画像リスト
    """

    # 画像を分割する際に端が切れてしまうのを防ぐために余白を追加する
    imgs = [cv2.copyMakeBorder(img, 0, size // 2, 0, size // 2, flg)
            for img in imgs]
    # 画像を分割しやすいように画像サイズを変更する
    v_size = imgs[0].shape[0] // size * size
    h_size = imgs[0].shape[1] // size * size
    imgs = [i[:v_size, :h_size] for i in imgs]
    # 画像の分割数を計算する
    v_split = imgs[0].shape[0] // size
    h_split = imgs[0].shape[1] // size
    # 画像を分割する
    out_imgs = []
    [[out_imgs.extend(np.vsplit(h_img, v_split))
      for h_img in np.hsplit(img, h_split)] for img in imgs]

    # バッチサイズの関係などで、画像の数を調整したい時はここで調整する
    # predict.pyなどで分割画像を復元したくなるので縦横の分割数も返す
    if(round_num > 0):
        round_len = len(out_imgs) // round_num * round_num
        return np.array(out_imgs[:round_len]), (v_split, h_split)
    else:
        return np.array(out_imgs), (v_split, h_split)


def imgs2x(imgs, flg=cv2.INTER_NEAREST):
    w, h = imgs[0].shape[:2]
    size = (w * 2, h * 2)
    return [cv2.resize(i, size, flg) for i in imgs]


def img2arr(imgs, norm=255, dtype=np.float32, gpu=-1):
    shape = imgs[0].shape
    w, h = shape[:2]
    if(len(shape) == 2):
        ch = 1
    else:
        ch = shape[2]

    if(gpu >= 0):
        return xp.array(imgs, dtype=dtype).reshape((-1, ch, w, h)) / norm
    else:
        return np.array(imgs, dtype=dtype).reshape((-1, ch, w, h)) / norm


def arr2img(arr, ch, size, norm=255, dtype=np.uint8):
    y = np.array(arr).reshape((-1, size, size, ch)) * 255
    return np.array(y, dtype=np.uint8)


def getLossfun(lossfun_str):
    if(lossfun_str.lower() == 'mse'):
        lossfun = F.mean_squared_error
    elif(lossfun_str.lower() == 'mae'):
        lossfun = F.mean_absolute_error
    elif(lossfun_str.lower() == 'abs'):
        lossfun = F.absolute_error
    elif(lossfun_str.lower() == 'se'):
        lossfun = F.squared_error
    elif(lossfun_str.lower() == 'softmax'):
        lossfun = F.softmax_cross_entropy
    else:
        lossfun = F.softmax_cross_entropy

    return lossfun


def getActFunc(actfunc_str):
    if(actfunc_str.lower() == 'relu'):
        actfunc = F.relu
    elif(actfunc_str.lower() == 'elu'):
        actfunc = F.elu
    elif(actfunc_str.lower() == 'c_relu'):
        actfunc = F.clipped_relu
    elif(actfunc_str.lower() == 'l_relu'):
        actfunc = F.leaky_relu
    elif(actfunc_str.lower() == 'sigmoid'):
        actfunc = F.sigmoid
    elif(actfunc_str.lower() == 'h_sigmoid'):
        actfunc = F.hard_sigmoid
    elif(actfunc_str.lower() == 'tanh'):
        actfunc = F.hard_sigmoid
    elif(actfunc_str.lower() == 's_plus'):
        actfunc = F.soft_plus
    else:
        actfunc = F.relu

    return actfunc


def getFilePath(folder, name, ext):
    if not os.path.isdir(folder):
        os.makedirs(folder)

    return os.path.join(folder, name + ext)

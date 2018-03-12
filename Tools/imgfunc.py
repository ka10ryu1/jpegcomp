#!/usr/bin/env python3
# -*-coding: utf-8 -*-
#
help = '画像処理に関する便利機能'
#

import os
import sys
import cv2
import numpy as np

try:
    import cupy as xp
except ImportError:
    print('not import cupy')

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


def random_rotate(imgs, num, level=[-10, 10], scale=1.2):

    def getCenter(img):
        return (img.shape[0]//2, img.shape[1]//2)

    out_imgs = []
    out_angle = []
    for n in range(num):
        for img in imgs:
            size = img.shape
            angle = np.random.randint(level[0], level[1])
            rot_mat = cv2.getRotationMatrix2D(getCenter(img), angle, scale)
            rot_img = cv2.warpAffine(img, rot_mat, size[:2], flags=cv2.INTER_CUBIC)
            out_imgs.append(rot_img[:size[0], :size[1]])
            out_angle.append(angle)

    return out_imgs, out_angle


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

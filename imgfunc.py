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


def blank(size, color, dtype=np.uint8):
    """
    単色画像を生成する
    [in]  size: 生成する画像サイズ [h,w,ch]（chがない場合は1を設定）
    [in]  color: 色（intでグレー、tupleでカラー）
    [in]  dtype: データ型
    [out] img:   生成した単色画像
    """

    # サイズに負数がある場合はエラー
    if np.min(size) < 0:
        print('[Error] size > 0: {0}'.format(size))
        print(fileFuncLine())
        exit()

    # サイズに縦横しか含まれていない場合はチャンネル追加
    if len(size) == 2:
        size = (size[0], size[1], 1)

    # 色がintの場合（0 < color < 255）
    if type(color) is int:
        img = np.zeros(size, dtype=dtype)
        if color < 0:
            color = 0
        elif color > 255:
            color = 255

        img.fill(color)
        return img

    # チャンネルが3じゃない時は3にする
    if size[2] != 3:
        size = (size[0], size[1], 3)

    img = np.zeros(size, dtype=dtype)
    img[:, :, :] = color
    return img


def isImgPath(name):
    """
    入力されたパスが画像か判定する
    [in]  name: 画像か判定したいパス
    [out] 画像ならTrue
    """

    if not type(name) is str:
        return False

    # cv2.imreadしてNoneが返ってきたら画像でないとする
    if cv2.imread(name) is not None:
        return True
    else:
        print('[{0}] is not Image'.format(name))
        print(fileFuncLine())
        return False


def encodeDecode(img, ch, quality=5):
    """
    入力された画像を圧縮する
    [in]  img:     入力画像
    [in]  ch:      圧縮画像のチャンネル数
    [in]  quality: 圧縮する品質 (1-100)
    [out] 圧縮画像
    """

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, encimg = cv2.imencode('.jpg', img, encode_param)
    if False == result:
        print(
            '[Error] {0}\n\tcould not encode image!'.format(fileFuncLine())
        )
        exit()

    return cv2.imdecode(encimg, getCh(ch))


def encodeDecodeN(imgs, ch, quality=5):
    """
    入力された画像リストを圧縮する
    [in]  imgs:    入力画像リスト
    [in]  ch:      出力画像リストのチャンネル数
    [in]  quality: 圧縮する品質 (1-100)
    [out] 出力画像リスト
    """

    return [encodeDecode(img, ch, quality) for img in imgs]


def cut(img, size):
    """
    画像を中心から任意のサイズで切り取る
    [in]  img:カットする画像
    [in]  size:カットするサイズ（正方形）
    [out] カットされた画像
    """

    if size <= 1:
        return img

    ch, cw = img.shape[0] // 2, img.shape[1] // 2
    return img[ch - size // 2:ch + size // 2, cw - size // 2:cw + size // 2]


def cutN(imgs, size, round_num=-1, flg=cv2.BORDER_REPLICATE):
    """
    画像リストの画像を中心から任意のサイズで切り取る
    [in]  img:カットする画像
    [in]  size:カットするサイズ（正方形）
    [out] カットされた画像
    """

    if size <= 1:
        return np.array(imgs)

    # 画像のカットを実行
    out_imgs = [cut(img, size) for img in imgs]
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
        return np.array(out_imgs[:round_len])
    else:
        return np.array(out_imgs)


def splitSQ(img, size, flg=cv2.BORDER_REPLICATE, array=True):
    """
    入力された画像を正方形に分割する
    [in]  img:   入力画像
    [in]  size:  正方形のサイズ [size x size]
    [in]  flg:   境界線のフラグ
    [out] imgs:  分割された正方形画像リスト
    [out] split: 縦横の分割情報
    """

    def arrayChk(x, flg):
        if flg:
            return np.array(x)
        else:
            return x

    def square(img):
        width = np.min(img.shape[:2])
        return img[:width, :width]

    h, w = img.shape[:2]
    split = (h // size, w // size)

    # sizeが負数だと分割しないでそのまま返す
    if size <= 1:

        return arrayChk([square(img)], array), (1, 1)

    # sizeが入力画像よりも大きい場合は分割しないでそのまま返す
    if split[0] == 0 or split[1] == 0:
        return arrayChk([square(img)], array), (1, 1)

    # 縦横の分割数を計算する
    if (h / size + w / size) > (h // size + w // size):
        # 画像を分割する際に端が切れてしまうのを防ぐために余白を追加する
        width = int(size * 0.2)
        img = cv2.copyMakeBorder(img, 0, width, 0, width, flg)
        # 画像を分割しやすいように画像サイズを変更する
        img = img[:split[0] * size, :split[1] * size]

    # 画像を分割する
    imgs_2d = [np.vsplit(i, split[0]) for i in np.hsplit(img, split[1])]
    imgs_1d = [x for l in imgs_2d for x in l]
    return arrayChk(imgs_1d, array), split


def splitSQN(imgs, size, round_num=-1, flg=cv2.BORDER_REPLICATE):
    """
    入力された画像リストを正方形に分割する
    imgsに格納されている画像はサイズが同じであること
    [in]  imgs:      入力画像リスト
    [in]  size:      正方形のサイズ（size x size）
    [in]  round_num: 丸める画像数
    [in]  flg:       境界線のフラグ
    [out] out_imgs:  分割されたnp.array形式の正方形画像リスト
    [out] split:     縦横の分割情報
    """

    if size <= 1:
        print('[Error] imgs[0].shape({0}), size({1})'.format(
            imgs[0].shape, size))
        print(fileFuncLine())
        exit()

    out_imgs = []
    split = []
    for img in imgs:
        i, s = splitSQ(img, size, flg, False)
        out_imgs.extend(i)
        split.extend(s)

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
        return np.array(out_imgs[:round_len]), (split[0], split[1])
    else:
        return np.array(out_imgs), (split[0], split[1])


def rotate(img, angle, scale):
    """
    画像を回転（反転）させる
    [in]  img:   回転させる画像
    [in]  angle: 回転させる角度
    [in]  scale: 拡大率
    [out] 回転させた画像
    """

    size = img.shape[:2]
    mat = cv2.getRotationMatrix2D((size[0] // 2, size[1] // 2), angle, scale)
    return cv2.warpAffine(img, mat, size, flags=cv2.INTER_CUBIC)


def rotateR(img, level=[-10, 10], scale=1.2):
    """
    ランダムに画像を回転させる
    [in]  img:   回転させる画像
    [in]  level: 回転させる角度の範囲
    [out] 回転させた画像
    [out] 回転させた角度
    """

    angle = np.random.randint(level[0], level[1])
    return rotate(img, angle, scale), angle


def rotateRN(imgs, num, level=[-10, 10], scale=1.2):
    """
    画像リストをランダムに画像を回転させる
    [in]  img:   回転させる画像
    [in]  num:   繰り返し数
    [in]  level: 回転させる角度の範囲
    [in]  scale: 拡大率
    [out] 回転させた画像リスト
    [out] 回転させた角度リスト
    """

    out_imgs = []
    out_angle = []
    for n in range(num):
        for img in imgs:
            i, a = rotateR(img, level, scale)
            out_imgs.append(i)
            out_angle.append(a)

    return np.array(out_imgs), np.array(out_angle)


def flip(img, num=2):
    """
    画像を回転させてデータ数を水増しする
    [in]  img:      入力画像
    [in]  num:      水増しする数（最大4倍）
    [out] out_imgs: 出力画像リスト
    """

    if(num < 1):
        return [img]

    # ベース
    out_imgs = [img.copy()]
    # 上下反転を追加
    f = cv2.flip(img, 0)
    out_imgs.append(f)
    if(num > 1):
        # 左右反転を追加
        f = cv2.flip(img, 1)
        out_imgs.append(f)

    if(num > 2):
        # 上下左右反転を追加
        f = cv2.flip(cv2.flip(img, 1), 0)
        out_imgs.append(f)

    return out_imgs


def flipN(imgs, num=2):
    """
    画像を回転させてデータ数を水増しする
    [in]  imgs:     入力画像リスト
    [in]  num:      水増しする数（最大4倍）
    [out] out_imgs: 出力画像リスト
    """

    if(num < 1):
        return np.array(imgs)

    # ベース
    out_imgs = imgs.copy()
    # 上下反転を追加
    f = [cv2.flip(i, 0) for i in imgs]
    out_imgs.extend(f)
    if(num > 1):
        # 左右反転を追加
        f = [cv2.flip(i, 1) for i in imgs]
        out_imgs.extend(f)

    if(num > 2):
        # 上下左右反転を追加
        f = [cv2.flip(cv2.flip(i, 1), 0) for i in imgs]
        out_imgs.extend(f)

    return np.array(out_imgs)


def whiteCheckN(imgs, val=245):
    """
    画像リストのうち、ほとんど白い画像を除去する
    [in] imgs: 判定する画像リスト
    [in] val:  除去するしきい値
    [out] ほとんど白い画像を除去した画像リスト
    """

    return np.array(
        [i for i in imgs if(val > np.sum(i) // (i.shape[0] * i.shape[1]))]
    )


def resize(img, rate, flg=cv2.INTER_NEAREST):
    """
    画像サイズを変更する
    [in] img:  N倍にする画像
    [in] rate: 倍率
    [in] flg:  N倍にする時のフラグ
    [out] N倍にされた画像リスト
    """

    if rate < 0:
        return img

    size = (int(img.shape[1] * rate),
            int(img.shape[0] * rate))
    return cv2.resize(img, size, flg)


def resizeP(img, pixel, flg=cv2.INTER_NEAREST):
    """
    画像サイズを変更する
    [in] img:   サイズを変更する画像
    [in] pixel: 短辺の幅
    [in] flg:   サイズを変更する時のフラグ
    [out] サイズを変更した画像リスト
    """

    r_img = resize(img, pixel / np.min(img.shape[:2]), flg)
    b_img = cv2.copyMakeBorder(
        r_img, 0, 2, 0, 2, cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )
    return b_img[:pixel, :pixel]


def resizeN(imgs, rate, flg=cv2.INTER_NEAREST):
    """
    画像リストの画像を全てサイズ変更する
    [in] img:  N倍にする画像
    [in] rate: 倍率
    [in] flg:  N倍にする時のフラグ
    [out] N倍にされた画像リスト
    """

    return np.array([resize(img, rate, flg) for img in imgs])


def size2x(imgs, flg=cv2.INTER_NEAREST):
    """
    画像のサイズを2倍にする
    [in] imgs: 2倍にする画像リスト
    [in] flg:  2倍にする時のフラグ
    [out] 2倍にされた画像リスト
    """

    return [resize(i, 2, flg) for i in imgs]


def paste(fg, bg, rot=0, x=0, y=0, mask_flg=True, rand_rot_flg=True, rand_pos_flg=True):
    """
    背景に前景を重ね合せる
    [in]  fg:         重ね合せる背景
    [in]  bg:         重ね合せる前景
    [in]  mask_flg:   マスク処理を大きめにするフラグ
    [in]  rand_rot_flg: 前景をランダムに回転するフラグ
    [in]  rand_pos_flg: 前景をランダムに配置するフラグ
    [out] 重ね合せた画像
    """

    # Load two images
    img1 = bg.copy()
    if rand_rot_flg:
        img2, rot = rotateR(fg, [-90, 90], 1.0)
    else:
        img2 = fg.copy()

    # I want to put logo on top-left corner, So I create a ROI
    w1, h1 = img1.shape[:2]
    w2, h2 = img2.shape[:2]
    if rand_pos_flg:
        x = np.random.randint(0, w1 - w2 + 1)
        y = np.random.randint(0, w1 - w2 + 1)

    roi = img1[x:x + w2, y:y + h2]

    def masked(img):
        if len(img.shape) < 3:
            return False
        elif img.shape[2] != 4:
            return False
        else:
            return True

    # Now create a mask of logo and create its inverse mask also
    if not masked(img2):
        mask = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(
            cv2.bitwise_not(mask), 10, 255, cv2.THRESH_BINARY
        )
    else:
        mask = img2[:, :, 3]

    ret, mask_inv = cv2.threshold(
        cv2.bitwise_not(mask), 200, 255, cv2.THRESH_BINARY
    )

    if mask_flg:
        kernel1 = np.ones((5, 5), np.uint8)
        kernel2 = np.ones((3, 3), np.uint8)
        mask_inv = cv2.dilate(mask_inv, kernel1, iterations=1)
        mask_inv = cv2.erode(mask_inv, kernel2, iterations=1)

    # Now black-out the area of logo in ROI
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

    # Take only region of logo from logo image.
    img2_fg = cv2.bitwise_and(img2, img2, mask=mask)

    # Put logo in ROI and modify the main image
    dst = cv2.add(img1_bg, img2_fg)
    img1[x:x + w2, y:y + h2] = dst
    return img1, (rot, x, y)


def arr2x(arr, flg=cv2.INTER_NEAREST):
    """
    行列を画像に変換し、サイズを2倍にする
    [in] arr: 2倍にする行列
    [in] flg: 2倍にする時のフラグ
    [out] 2倍にされた行列
    """

    imgs = arr2imgs(arr)
    return imgs2arr(size2x(imgs, flg))


def arrNx(arr, rate, flg=cv2.INTER_NEAREST):
    """
    行列を画像に変換し、サイズをN倍にする
    [in] arr:  N倍にする行列
    [in] rate: 倍率
    [in] flg:  N倍にする時のフラグ
    [out] N倍にされた行列
    """

    if(len(arr.shape) == 3):
        img = arr2img(arr)
        return img2arr(resize(img, rate, flg))

    if(len(arr.shape) == 4):
        imgs = arr2imgs(arr)
        return imgs2arr(resizeN(imgs, rate, flg))


def img2arr(img, norm=255, dtype=np.float32, gpu=-1):
    try:
        w, h, _ = img.shape
    except:
        w, h = img.shape[:2]

    if(gpu >= 0):
        return xp.array(img, dtype=dtype).reshape((-1, w, h)) / norm
    else:
        return np.array(img, dtype=dtype).reshape((-1, w, h)) / norm


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


def arr2img(arr, norm=255, dtype=np.uint8):
    ch, size = arr.shape[-3], arr.shape[-2]
    y = np.array(arr).reshape((size, size, ch)) * 255
    return np.array(y, dtype=np.uint8)


def arr2imgs(arr, norm=255, dtype=np.uint8):
    """
    Chainerの出力をOpenCVで可視化するために変換する
    [in]  arr:   Chainerから出力された行列
    [in]  norm:  正規化をもとに戻す数（255であれば、0-1を0-255に変換する）
    [in]  dtype: 変換するデータタイプ
    [out] OpenCV形式の画像に変換された行列
    """

    try:
        ch, size = arr.shape[1], arr.shape[2]
    except:
        print('[ERROR] input data is not img arr')
        print(fileFuncLine())
        exit()

    y = np.array(arr).reshape((-1, size, size, ch)) * 255
    return np.array(y, dtype=np.uint8)

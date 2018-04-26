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


def blank(size, color, dtype=np.uint8, min_val=0, max_val=255):
    """
    単色画像を生成する
    [in]  size: 生成する画像サイズ [h,w,ch]（chがない場合は1を設定）
    [in]  color: 色（intでグレー、tupleでカラー）
    [in]  dtype: データ型
    [in]  min_val: 色の最小値
    [in]  max_val: 色の最大値
    [out] img:   生成した単色画像
    """

    # サイズに負数がある場合はエラー
    if np.min(size) < 0:
        print('[Error] size < 0: {0}'.format(size))
        print(fileFuncLine())
        exit(1)

    # サイズに縦横しか含まれていない場合はチャンネル追加
    if len(size) == 2:
        size = (size[0], size[1], 1)

    # 色がintの場合はグレースケールとして塗りつぶす
    # 0 < color < 255の範囲にない場合は丸める
    if type(color) is int:
        img = np.zeros(size, dtype=dtype)
        if color < min_val:
            color = min_val
        elif color > max_val:
            color = max_val

        img.fill(color)
        return img

    # チャンネルが3じゃない時は3にする
    if size[2] != 3:
        size = (size[0], size[1], 3)

    img = np.zeros(size, dtype=dtype)
    img[:, :, :] = color
    return img


def isImgPath(name, silent=False):
    """
    入力されたパスが画像か判定する
    [in]  name:   画像か判定したいパス
    [in]  silent: cv2.imread失敗時にエラーを表示させない場合はTrue
    [out] 画像ならTrue
    """

    if not type(name) is str:
        return False

    # cv2.imreadしてNoneが返ってきたら画像でないとする
    if cv2.imread(name) is not None:
        return True
    else:
        if not silent:
            print('[{0}] is not Image'.format(name))
            print(fileFuncLine())

        return False


def encodeDecode(img, ch, quality=5, ext='.jpg'):
    """
    入力された画像を圧縮する
    ※詳細はencodeDecodeNとほぼ同じなので省略
    """

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, encimg = cv2.imencode(ext, img, encode_param)
    if False == result:
        print(
            '[Error] {0}\n\timage encode failed!'.format(fileFuncLine())
        )
        exit(1)

    return cv2.imdecode(encimg, getCh(ch))


def encodeDecodeN(imgs, ch, quality=5, ext='.jpg'):
    """
    入力された画像リストを圧縮する
    [in]  imgs:    入力画像リスト
    [in]  ch:      出力画像リストのチャンネル数
    [in]  quality: 圧縮する品質 (1-100)
    [in]  ext:     圧縮する拡張子
    [out] 圧縮画像リスト
    """

    return [encodeDecode(img, ch, quality, ext) for img in imgs]


def cut(img, size=-1):
    """
    画像を中心から任意のサイズで切り取る
    ※詳細はcutNとほぼ同じなので省略
    """

    # カットするサイズの半分を計算する
    if size <= 1:
        # サイズが1以下の場合、imgの短辺がカットするサイズになる
        half = np.min(img.shape[:2])//2
    else:
        half = size // 2

    # 画像の中心位置を計算
    ch, cw = img.shape[0] // 2, img.shape[1] // 2
    return img[ch - half:ch + half, cw - half:cw + half]


def cutN(imgs, size=-1, round_num=-1):
    """
    画像リストの画像を中心から任意のサイズで切り取る
    [in]  img:       カットする画像
    [in]  size:      カットするサイズ（正方形）
    [in]  round_num: 丸める数
    [out] カットされた画像リスト
    """

    # 画像のカットを実行
    out_imgs = [cut(img, size) for img in imgs]
    # 切り捨てたい数よりも画像数が少ないと0枚になってしまうので注意
    if(round_num > len(out_imgs)):
        round_num = -1

    # バッチサイズの関係などで、画像の数を調整したい時はここで調整する
    # predict.pyなどで分割画像を復元したくなるので縦横の分割数も返す
    if(round_num > 0):
        round_len = len(out_imgs) // round_num * round_num
        return np.array(out_imgs[:round_len])
    else:
        return np.array(out_imgs)


def splitSQ(img, size, flg=cv2.BORDER_REPLICATE, w_rate=0.2, array=True):
    """
    入力された画像を正方形に分割する
    ※詳細はsplitSQNとほぼ同じなので省略
    """

    def arrayChk(x, to_arr):
        # np.array (True)にするか、list (False)にするか選択する
        if to_arr:
            return np.array(x)
        else:
            return x

    # sizeが負数だと分割しないでそのまま返す
    if size <= 1:
        return arrayChk(cutN(img), array), (1, 1)

    h, w = img.shape[:2]
    split = (h // size, w // size)

    # sizeが入力画像よりも大きい場合は分割しないでそのまま返す
    if split[0] == 0 or split[1] == 0:
        return arrayChk([cut(img)], array), (1, 1)

    # 縦横の分割数を計算する
    if (h / size + w / size) > (h // size + w // size):
        # 画像を分割する際に端が切れてしまうのを防ぐために余白を追加する
        width = int(size * w_rate)
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

    out_imgs = []
    split = []
    for img in imgs:
        i, s = splitSQ(img, size, flg, False)
        out_imgs.extend(i)
        split.extend(s)

    # 切り捨てたい数よりも画像数が少ないと0枚になってしまうので注意
    if(round_num > len(out_imgs)):
        round_num = -1

    # バッチサイズの関係などで、画像の数を調整したい時はここで調整する
    # predict.pyなどで分割画像を復元したくなるので縦横の分割数も返す
    if(round_num > 0):
        round_len = len(out_imgs) // round_num * round_num
        return np.array(out_imgs[:round_len]), (split[0], split[1])
    else:
        return np.array(out_imgs), (split[0], split[1])


def rotate(img, angle, scale, border=(0, 0, 0)):
    """
    画像を回転（反転）させる
    [in]  img:    回転させる画像
    [in]  angle:  回転させる角度
    [in]  scale:  拡大率
    [in]  border: 回転時の画像情報がない場所を埋める色
    [out] 回転させた画像
    """

    size = img.shape[:2]
    mat = cv2.getRotationMatrix2D((size[0] // 2, size[1] // 2), angle, scale)
    return cv2.warpAffine(img, mat, size, flags=cv2.INTER_CUBIC, borderValue=border)


def rotateR(img, level=[-10, 10], scale=1.2, border=(0, 0, 0)):
    """
    ランダムに画像を回転させる
    ※詳細はrotateRNとほぼ同じなので省略
    """

    angle = np.random.randint(level[0], level[1])
    return rotate(img, angle, scale, border), angle


def rotateRN(imgs, num, level=[-10, 10], scale=1.2, border=(0, 0, 0)):
    """
    画像リストをランダムに画像を回転させる
    [in]  img:   回転させる画像
    [in]  num:   繰り返し数
    [in]  level: 回転させる角度の範囲
    [in]  scale: 拡大率
    [in]  border: 回転時の画像情報がない場所を埋める色
    [out] 回転させた画像リスト
    [out] 回転させた角度リスト
    """

    out_imgs = []
    out_angle = []
    for n in range(num):
        for img in imgs:
            i, a = rotateR(img, level, scale, border)
            out_imgs.append(i)
            out_angle.append(a)

    return np.array(out_imgs), np.array(out_angle)


def flip(img, num=2):
    """
    画像を回転させてデータ数を水増しする
    ※詳細はflipNとほぼ同じなので省略
    """

    if(num < 1):
        return [img]

    horizontal = 0
    vertical = 1
    # ベース
    out_imgs = [img.copy()]
    # 上下反転を追加
    f = cv2.flip(img, horizontal)
    out_imgs.append(f)
    if(num > 1):
        # 左右反転を追加
        f = cv2.flip(img, vertical)
        out_imgs.append(f)

    if(num > 2):
        # 上下左右反転を追加
        f = cv2.flip(cv2.flip(img, horizontal), vertical)
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

    horizontal = 0
    vertical = 1
    # ベース
    out_imgs = imgs.copy()
    # 上下反転を追加
    f = [cv2.flip(i, horizontal) for i in imgs]
    out_imgs.extend(f)
    if(num > 1):
        # 左右反転を追加
        f = [cv2.flip(i, vertical) for i in imgs]
        out_imgs.extend(f)

    if(num > 2):
        # 上下左右反転を追加
        f = [cv2.flip(cv2.flip(i, vertical), horizontal) for i in imgs]
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

    rate = 2
    return [resize(i, rate, flg) for i in imgs]


def paste(fg, bg, rot=0, x=0, y=0, mask_flg=True, rand_rot_flg=True, rand_pos_flg=True):
    """
    背景に前景を重ね合せる
    [in]  fg:           重ね合せる前景
    [in]  bg:           重ね合せる背景
    [in]  rot:          重ね合わせ時の前景回転角
    [in]  x:            重ね合わせ時の前景x位置
    [in]  y:            重ね合わせ時の前景y位置
    [in]  mask_flg:     マスク処理を大きめにするフラグ
    [in]  rand_rot_flg: 前景をランダムに回転するフラグ
    [in]  rand_pos_flg: 前景をランダムに配置するフラグ
    [out] 重ね合せた画像
    http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_core/py_image_arithmetics/py_image_arithmetics.html#bitwise-operations
    """

    # 画素の最大値
    max_val = 255

    # Load two images
    img1 = bg.copy()
    angle = [-90, 90]  # ランダム回転の範囲
    scale = 1.0  # 画像の拡大率
    white = (max_val, max_val, max_val)
    if rand_rot_flg:
        # ランダムに回転
        img2, rot = rotateR(fg, angle, scale, white)
        print('rot', rot)
    else:
        # 任意の角度で回転
        img2 = rotate(fg, rot, scale, white)

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
        thresh = 10
        mask = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(
            cv2.bitwise_not(mask), thresh, max_val, cv2.THRESH_BINARY
        )
    else:
        mask = img2[:, :, 3]

    thresh = 200
    ret, mask_inv = cv2.threshold(
        cv2.bitwise_not(mask), thresh, max_val, cv2.THRESH_BINARY
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

    return imgs2arr(size2x(arr2imgs(arr), flg))


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
    """
    入力画像をChainerで利用するために変換する
    ※詳細はimgs2arrとほぼ同じなので省略
    """

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
    """
    Chainerの出力をOpenCVで可視化するために変換する入力（単画像用）
    ※詳細はarr2imgsとほぼ同じなので省略
    """

    try:
        ch, h, w = arr.shape
    except:
        h, w = arr.shape
        ch = 1

    y = np.array(arr).reshape((h, w, ch)) * norm
    return np.array(y, dtype=dtype)


def arr2imgs(arr, norm=255, dtype=np.uint8):
    """
    Chainerの出力をOpenCVで可視化するために変換する（画像リスト用）
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
        exit(1)

    y = np.array(arr).reshape((-1, size, size, ch)) * norm
    return np.array(y, dtype=dtype)

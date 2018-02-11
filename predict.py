#!/usr/bin/env python3
# -*-coding: utf-8 -*-
#
help = 'モデルとモデルパラメータを利用して推論実行する'
#

import os
import cv2
import json
import time
import argparse
import numpy as np

import chainer
import chainer.links as L
from chainer.cuda import to_cpu

from Lib.network2 import JC
import Lib.imgfunc as IMG
import Tools.func as F


def command():
    parser = argparse.ArgumentParser(description=help)
    parser.add_argument('model',
                        help='使用する学習済みモデル')
    parser.add_argument('param',
                        help='使用するモデルパラメータ')
    parser.add_argument('jpeg', nargs='+',
                        help='使用する画像のパス')
    parser.add_argument('--img_size', '-s', type=int, default=32,
                        help='生成される画像サイズ [default: 32 pixel]')
    parser.add_argument('--quality', '-q', type=int, default=5,
                        help='画像の圧縮率 [default: 5]')
    parser.add_argument('--batch', '-b', type=int, default=100,
                        help='ミニバッチサイズ [default: 100]')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID [default -1]')
    parser.add_argument('--out_path', '-o', default='./result/',
                        help='生成物の保存先[default: ./result/]')
    return parser.parse_args()


def getModelParam(path):
    """
    jsonで記述されたモデルパラメータ情報を読み込む
    [in]  path:        jsonファイルのパス
    [out] d['unut']:   中間層のユニット数
    [out] d['img_ch']: 画像のチャンネル数
    [out] d['layer']:  ネットワーク層の数
    [out] af1:         活性化関数(1)
    [out] af2:         活性化関数(2)
    """

    print('model param:', path)
    try:
        with open(path, 'r') as f:
            d = json.load(f)

    except:
        import traceback
        traceback.print_exc()
        print(F.fileFuncLine())
        exit()

    af1 = IMG.getActfun(d['actfun_1'])
    af2 = IMG.getActfun(d['actfun_2'])
    return d['unit'], d['img_ch'], d['layer'], d['shuffle_rate'], af1, af2


def predict(model, args, img, ch, val):
    """
    推論実行メイン部
    [in]  model:  推論実行に使用するモデル
    [in]  args:   実行時のオプション引数情報
    [in]  img:    入力画像
    [in]  ch:     入力画像のチャンネル数
    [in]  val:    画像保存時の連番情報
    [out] img:推論実行で得られた生成画像
    """

    org_size = img.shape
    # 入力画像を圧縮して劣化させる
    comp = IMG.encodeDecode([img], IMG.getCh(ch), args.quality)
    # 比較のため圧縮画像を保存する
    if(val >= 0):
        cv2.imwrite(
            F.getFilePath(args.out_path, 'comp-' +
                          str(val * 10).zfill(3), '.jpg'),
            comp[0]
        )

    # 入力画像を分割する
    comp, size = IMG.split(comp, args.img_size)
    imgs = []

    st = time.time()

    # バッチサイズごとに学習済みモデルに入力して画像を生成する
    for i in range(0, len(comp), args.batch):
        x = IMG.imgs2arr(comp[i:i + args.batch], gpu=args.gpu)
        y = model.predictor(x)
        y = to_cpu(y.array)
        y = IMG.arr2imgs(y, ch, args.img_size * 2)
        imgs.extend(y)

    print('exec time: {0:.2f}[s]'.format(time.time() - st))

    # 生成された画像を結合する
    buf = [np.vstack(imgs[i * size[0]: (i + 1) * size[0]])
           for i in range(size[1])]
    img = np.hstack(buf)
    # 生成された画像は入力画像の2倍の大きさになっているので縮小する
    h = 0.5
    half_size = (int(img.shape[1] * h), int(img.shape[0] * h))
    flg = cv2.INTER_NEAREST
    img = cv2.resize(img, half_size, flg)
    img = img[:org_size[0], :org_size[1]]
    # 生成結果を保存する
    if(val >= 0):
        name = F.getFilePath(args.out_path, 'comp-' +
                             str(val * 10 + 1).zfill(3), '.jpg')
        print('save:', name)
        cv2.imwrite(name, img)

    return img


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
        print(F.fileFuncLine())
        return False


def checkModelType(path):
    """
    入力されたパスが.modelか.snapshotかそれ以外か判定し、
    load_npzのpathを設定する
    [in]  path:      入力されたパス
    [out] load_path: load_npzのpath
    """

    # 拡張子を正とする
    name, ext = os.path.splitext(os.path.basename(path))
    load_path = ''
    if(ext == '.model'):
        print('model read:', path)
    elif(ext == '.snapshot'):
        print('snapshot read', path)
        load_path = 'updater/model:main/'
    else:
        print('model read error')
        print(F.fileFuncLine())
        exit()

    return load_path


def main(args):
    # jsonファイルから学習モデルのパラメータを取得する
    unit, ch, layer, sr, af1, af2 = getModelParam(args.param)
    # 学習モデルの出力画像のチャンネルに応じて画像を読み込む
    ch_flg = IMG.getCh(ch)
    imgs = [cv2.imread(name, ch_flg) for name in args.jpeg if isImage(name)]
    # 学習モデルを生成する
    model = L.Classifier(
        JC(n_unit=unit, n_out=ch, layer=layer,
           rate=sr, actfun_1=af1, actfun_2=af2)
    )
    # load_npzのpath情報を取得する
    load_path = checkModelType(args.model)
    # 学習済みモデルの読み込み
    try:
        chainer.serializers.load_npz(args.model, model, path=load_path)
    except:
        import traceback
        traceback.print_exc()
        print(F.fileFuncLine())
        exit()

    # GPUの設定
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    # 学習モデルを入力画像ごとに実行する
    with chainer.using_config('train', False):
        imgs = [predict(model, args, img, ch, i)
                for i, img in enumerate(imgs)]

    # 生成結果の表示
    for i in imgs:
        cv2.imshow('test', i)
        cv2.waitKey()


if __name__ == '__main__':
    args = command()
    F.argsPrint(args)
    main(args)

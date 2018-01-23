#!/usr/bin/env python3
# -*-coding: utf-8 -*-
#
help = 'スナップショットを利用した画像の生成'
#

import os
import cv2
import json
import argparse
import numpy as np

import chainer
import chainer.links as L
from chainer.cuda import to_cpu

from Lib.network import JC
import Lib.myfunc as M


def command():
    parser = argparse.ArgumentParser(description=help)
    parser.add_argument('model',
                        help='使用する学習済みモデル')
    parser.add_argument('param',
                        help='使用するモデルパラメータ')
    parser.add_argument('jpeg', nargs='+',
                        help='使用する画像のパス')
    parser.add_argument('--img_size', '-s', type=int, default=32,
                        help='生成される画像サイズ（default: 32 pixel）')
    parser.add_argument('--quality', '-q', type=int, default=5,
                        help='画像の圧縮率（default: 5）')
    parser.add_argument('--batch', '-b', type=int, default=100,
                        help='ミニバッチサイズ (default: 100)')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (default -1)')
    parser.add_argument('--out_path', '-o', default='./result/',
                        help='生成物の保存先(default: ./result/)')
    return parser.parse_args()


def getModelParam(path):
    try:
        with open(path, 'r') as f:
            d = json.load(f)

    except:
        import traceback
        traceback.print_exc()
        print(M.fileFuncLine())
        exit()

    af1 = M.getActfun(d['actfun_1'])
    af2 = M.getActfun(d['actfun_2'])
    return d['unit'], d['img_ch'], d['layer'], af1, af2


def predict(model, args, img, ch, ch_flg, val):

    # 入力画像を圧縮して劣化させる
    comp = M.imgEncodeDecode([img], ch_flg, args.quality)
    # 比較のため圧縮画像を保存する
    cv2.imwrite(
        M.getFilePath(args.out_path, 'comp-' + str(val * 10).zfill(3), '.jpg'),
        comp[0]
    )

    # 入力画像を分割する
    comp, size = M.imgSplit(comp, args.img_size)
    imgs = []
    # バッチサイズごとに学習済みモデルに入力して画像を生成する
    for i in range(0, len(comp), args.batch):
        x = M.img2arr(comp[i:i + args.batch], gpu=args.gpu)
        y = model.predictor(x)
        y = to_cpu(y.array)
        y = M.arr2img(y, ch, args.img_size * 2)
        imgs.extend(y)

    # 生成された画像を結合する
    buf = [np.vstack(imgs[i * size[0]: (i + 1) * size[0]])
           for i in range(size[1])]
    img = np.hstack(buf)
    # 生成された画像は入力画像の2倍の大きさになっているので縮小する
    h = 0.5
    half_size = (int(img.shape[1] * h), int(img.shape[0] * h))
    flg = cv2.INTER_NEAREST
    img = cv2.resize(img, half_size, flg)
    # 生成結果を保存する
    cv2.imwrite(
        M.getFilePath(args.out_path, 'comp-' +
                      str(val * 10 + 1).zfill(3), '.jpg'),
        img
    )
    return img


def isImage(name):
    if cv2.imread(name) is not None:
        return True
    else:
        print('[{0}] is not Image'.format(name))
        print(M.fileFuncLine())
        return False


def checkModelType(path):
    name, ext = os.path.splitext(os.path.basename(path))
    load_path = ''
    if(ext == '.model'):
        print('model read')
    elif(ext == '.snapshot'):
        print('snapshot read')
        load_path = 'updater/model:main/'
    else:
        print('model read error')
        print(M.fileFuncLine())
        exit()

    return load_path


def main(args):
    # jsonファイルから学習モデルのパラメータを取得する
    unit, ch, layer, af1, af2 = getModelParam(args.param)
    # 学習モデルの出力画像のチャンネルに応じて画像を読み込む
    ch_flg = M.getCh(ch)
    imgs = [cv2.imread(name, ch_flg) for name in args.jpeg if isImage(name)]
    # 学習モデルを生成する
    model = L.Classifier(
        JC(n_unit=unit, n_out=ch, layer=layer, actfun_1=af1, actfun_2=af2)
    )

    load_path = checkModelType(args.model)

    try:
        chainer.serializers.load_npz(args.model, model, path=load_path)
    except:
        import traceback
        traceback.print_exc()
        print(M.fileFuncLine())
        exit()

    # GPUの設定
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    # 学習モデルを入力画像ごとに実行する
    imgs = [predict(model, args, img, ch, ch_flg, i)
            for i, img in enumerate(imgs)]
    # 生成結果の表示
    for i in imgs:
        cv2.imshow('test', i)
        cv2.waitKey()


if __name__ == '__main__':
    args = command()
    M.argsPrint(args)
    main(args)

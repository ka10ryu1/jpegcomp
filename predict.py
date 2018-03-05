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

import Lib.imgfunc as IMG
import Tools.func as F
from concat_3_images import concat3Images


def command():
    parser = argparse.ArgumentParser(description=help)
    parser.add_argument('model',
                        help='使用する学習済みモデル')
    parser.add_argument('param',
                        help='使用するモデルパラメータ')
    parser.add_argument('jpeg', nargs='+',
                        help='使用する画像のパス')
    # parser.add_argument('--img_size', '-s', type=int, default=32,
    #                     help='生成される画像サイズ [default: 32 pixel]')
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
    ch = d['shape'][0]
    size = d['shape'][1]
    return \
        d['network'], d['unit'], ch, size, \
        d['layer_num'], d['shuffle_rate'], af1, af2


def encDecWrite(img, ch, quality, out_path='./result', val=-1):
    # 入力画像を圧縮して劣化させる
    comp = IMG.encodeDecode([img], IMG.getCh(ch), quality)
    # 比較のため圧縮画像を保存する
    if(val >= 0):
        path = F.getFilePath(out_path, 'comp-' + str(val * 10).zfill(3), '.jpg')
        cv2.imwrite(path, comp[0])

    return comp[0]


def predict(model, data, batch, org_shape, gpu):
    """
    推論実行メイン部
    [in]  model:     推論実行に使用するモデル
    [in]  data:      分割（IMG.split）されたもの
    [in]  batch:     バッチサイズ
    [in]  org_shape: 分割前のshape
    [in]  gpu:       GPU ID
    [out] img:       推論実行で得られた生成画像
    """

    comp, size = data
    imgs = []
    st = time.time()
    # バッチサイズごとに学習済みモデルに入力して画像を生成する
    for i in range(0, len(comp), batch):
        x = IMG.imgs2arr(comp[i:i + batch], gpu=gpu)
        y = model.predictor(x)
        imgs.extend(IMG.arr2imgs(to_cpu(y.array)))

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
    img = img[:org_shape[0], :org_shape[1]]

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
    net, unit, ch, size, layer, sr, af1, af2 = getModelParam(args.param)
    # 学習モデルを生成する
    if net == 0:
        from Lib.network import JC_DDUU as JC
    else:
        from Lib.network2 import JC_UDUD as JC

    model = L.Classifier(
        JC(n_unit=unit, n_out=ch, layer=layer,
           rate=sr, actfun_1=af1, actfun_2=af2)
    )
    # load_npzのpath情報を取得し、学習済みモデルを読み込む
    load_path = checkModelType(args.model)
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
    ch_flg = IMG.getCh(ch)
    org_imgs = [cv2.imread(name, ch_flg) for name in args.jpeg if isImage(name)]
    ed_imgs = [encDecWrite(img, ch, args.quality, args.out_path, i)
               for i, img in enumerate(org_imgs)]
    imgs = []
    with chainer.using_config('train', False):
        for i, ei in enumerate(ed_imgs):
            img = predict(model, IMG.split([ei], size), args.batch, ei.shape, args.gpu)
            # 生成結果を保存する
            name = F.getFilePath(args.out_path, 'comp-' +
                                 str(i * 10 + 1).zfill(3), '.jpg')
            print('save:', name)
            cv2.imwrite(name, img)
            imgs.append(img)

    # オリジナル、高圧縮、推論実行結果を連結して保存・表示する
    c3i = [concat3Images([i, j, k], 50, 333, ch, 1)
           for i, j, k in zip(org_imgs, ed_imgs, imgs)]
    path = [F.getFilePath(args.out_path, 'concat-' + str(i * 10).zfill(3), '.jpg')
            for i in range(len(c3i))]
    [cv2.imwrite(p, i) for p, i in zip(path, c3i)]
    for i in c3i:
        cv2.imshow('test', i)
        cv2.waitKey()


if __name__ == '__main__':
    args = command()
    F.argsPrint(args)
    main(args)

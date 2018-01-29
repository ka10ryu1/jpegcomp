#!/usr/bin/env python3
# -*-coding: utf-8 -*-
#
help = '複数のsnapshotoとひとつのモデルパラメータを利用してsnapshotの推移を可視化する'
#

import os
import cv2
import argparse
import numpy as np

import chainer
import chainer.links as L

from Lib.network import JC
import Lib.imgfunc as IMG
import Tools.func as F
from predict import getModelParam, predict, isImage, checkModelType


def command():
    parser = argparse.ArgumentParser(description=help)
    parser.add_argument('snapshot_and_json',
                        help='使用するスナップショットとモデルパラメータのあるフォルダ')
    parser.add_argument('jpeg', nargs='+',
                        help='使用する画像のパス')
    parser.add_argument('--img_size', '-is', type=int, default=32,
                        help='生成される画像サイズ（default: 32 pixel）')
    parser.add_argument('--quality', '-q', type=int, default=5,
                        help='画像の圧縮率（default: 5）')
    parser.add_argument('--batch', '-b', type=int, default=100,
                        help='ミニバッチサイズ (default: 100)')
    parser.add_argument('--image_num', '-n', type=int, default=10,
                        help='切り出す画像数 (default: 10)')
    parser.add_argument('--random_seed', '-rs', type=int, default=25,
                        help='乱数シード（default: 25, random: -1）')
    parser.add_argument('--img_rate', '-r', type=int, default=4,
                        help='画像サイズの倍率（default: 4）')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (default -1)')
    parser.add_argument('--out_path', '-o', default='./result/',
                        help='生成物の保存先(default: ./result/)')
    return parser.parse_args()


def main(args):
    snapshot_path = []
    param = ''
    for f in os.listdir(args.snapshot_and_json):
        name, ext = os.path.splitext(os.path.basename(f))
        full_path = os.path.join(args.snapshot_and_json, f)
        print(f, name, ext)
        if('.snapshot' in ext):
            snapshot_path.append(full_path)
        elif('.json' in ext):
            param = full_path

    snapshot_path = sorted([x for x in snapshot_path
                            if os.path.isfile(x)], key=os.path.getmtime)
    # jsonファイルから学習モデルのパラメータを取得する
    unit, ch, layer, af1, af2 = getModelParam(param)
    # 学習モデルの出力画像のチャンネルに応じて画像を読み込む
    ch_flg = IMG.getCh(ch)
    imgs = [cv2.imread(jpg, ch_flg) for jpg in args.jpeg if isImage(jpg)]
    imgs, size = IMG.split(imgs, args.img_size)
    imgs = np.array(IMG.whiteCheck(imgs))
    if(args.random_seed >= 0):
        np.random.seed(args.random_seed)

    shuffle = np.random.permutation(range(len(imgs)))
    img = np.vstack(imgs[shuffle[:args.image_num]])

    # 学習モデルを生成する
    model = L.Classifier(JC(
        n_unit=unit,
        n_out=ch,
        layer=layer,
        actfun_1=af1,
        actfun_2=af2
    ))

    out_imgs = [img]
    for s in snapshot_path:

        # load_npzのpath情報を取得する
        load_path = checkModelType(s)
        # 学習済みモデルの読み込み
        try:
            chainer.serializers.load_npz(s, model, path=load_path)
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
        out_imgs.append(predict(model, args, img, ch, -1))

    # 生成結果の表示
    img = np.hstack(out_imgs)
    w, h = img.shape[:2]
    img = cv2.resize(img, (h * args.img_rate, w * args.img_rate),
                     cv2.INTER_NEAREST)
    cv2.imshow('predict some snapshots', img)
    cv2.waitKey()
    cv2.imwrite(F.getFilePath(args.out_path, 'snapshots.jpg'), img)


if __name__ == '__main__':
    args = command()
    F.argsPrint(args)
    main(args)

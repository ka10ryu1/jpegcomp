#!/usr/bin/env python3
# -*-coding: utf-8 -*-
#
help = '複数のsnapshotoとひとつのモデルパラメータを利用してsnapshotの推移を可視化する'
#

import logging
# basicConfig()は、 debug()やinfo()を最初に呼び出す"前"に呼び出すこと
logging.basicConfig(format='%(message)s')
logging.getLogger('Tools').setLevel(level=logging.INFO)

import os
import cv2
import argparse
import numpy as np

import chainer
import chainer.links as L

import Tools.imgfunc as IMG
import Tools.getfunc as GET
import Tools.func as F
from predict import encDecWrite, predict


def command():
    parser = argparse.ArgumentParser(description=help)
    parser.add_argument('snapshot_and_json',
                        help='使用するスナップショットとモデルパラメータのあるフォルダ')
    parser.add_argument('jpeg', nargs='+',
                        help='使用する画像のパス')
    parser.add_argument('--quality', '-q', type=int, default=5,
                        help='画像の圧縮率 [default: 5]')
    parser.add_argument('--batch', '-b', type=int, default=20,
                        help='ミニバッチサイズ [default: 20]')
    parser.add_argument('--img_num', '-n', type=int, default=10,
                        help='切り出す画像数 [default: 10]')
    parser.add_argument('--random_seed', '-rs', type=int, default=25,
                        help='乱数シード [default: 25, random: -1]')
    parser.add_argument('--img_rate', '-r', type=float, default=1,
                        help='画像サイズの倍率 [default: 1.0]')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID [default -1]')
    parser.add_argument('--out_path', '-o', default='./result/',
                        help='生成物の保存先[default: ./result/]')
    args = parser.parse_args()
    F.argsPrint(args)
    return args


def getSnapshotAndParam(folder):
    """
    入力フォルダからスナップショットとモデルパラメータのパスを取得する
    スナップショットは作成された日時順にソートする
    [in]  folder:        スナップショットとモデルパラメータのあるフォルダ
    [out] snapshot_path: スナップショットのパスのリスト
    [out] param:         モデルパラメータのパス
    """

    snapshot_path = []
    param_path = ''
    for f in os.listdir(folder):
        name, ext = os.path.splitext(os.path.basename(f))
        full_path = os.path.join(folder, f)
        # print(f, name, ext)
        if('.snapshot' in ext):
            snapshot_path.append(full_path)
        elif('.json' in ext):
            param_path = full_path

    snapshot_path = sorted([x for x in snapshot_path
                            if os.path.isfile(x)], key=os.path.getmtime)

    return snapshot_path, param_path


def getImage(jpg_path, ch, img_size, img_num, seed):
    """
    画像を読み込んでランダムに取得して連結する
    [in]  jpg_path: 入力画像のパス
    [in]  ch:       画像を読み込むチャンネル数
    [in]  img_size: 画像を分割するサイズ
    [in]  img_num:  使用する画像の数
    [in]  seed:     乱数シード
    [out] 連結された画像（縦長）
    """

    ch_flg = IMG.getCh(ch)
    # 画像を読み込み
    imgs = [cv2.imread(jpg, ch_flg) for jpg in jpg_path if IMG.isImgPath(jpg)]
    # 画像を分割し
    imgs, size = IMG.splitSQN(imgs, img_size)
    # ほとんど白い画像を除去し
    imgs = np.array(IMG.whiteCheckN(imgs))
    if(seed >= 0):
        np.random.seed(seed)

    # ランダムに取得する
    shuffle = np.random.permutation(range(len(imgs)))

    return np.vstack(imgs[shuffle[:img_num]])


def stackImages(imgs, rate):
    """
    推論実行で得られた 画像のリストを結合してサイズを調整する
    [in]  imgs:    入力画像リスト
    [in]  rate: リサイズする倍率
    [out] 結合画像
    """

    img = np.hstack(
        [cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if len(img.shape) < 3 else img
         for img in imgs]
    )
    return IMG.resize(img, rate)


def main(args):
    # スナップショットとモデルパラメータのパスを取得する
    snapshot_path, param = getSnapshotAndParam(args.snapshot_and_json)
    # jsonファイルから学習モデルのパラメータを取得する
    net, unit, ch, size, layer, sr, af1, af2 = GET.modelParam(param)
    # 推論実行するために画像を読み込んで結合する
    img = getImage(args.jpeg, ch, size, args.img_num, args.random_seed)
    # 学習モデルを生成する
    if net == 0:
        from Lib.network import JC_DDUU as JC
    else:
        from Lib.network2 import JC_UDUD as JC

    model = L.Classifier(
        JC(n_unit=unit, n_out=1, layer=layer, rate=sr, actfun1=af1, actfun2=af2)
    )
    out_imgs = [img]
    for s in snapshot_path:
        # load_npzのpath情報を取得する
        load_path = F.checkModelType(s)
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
        else:
            model.to_intel64()

        # 学習モデルを入力画像ごとに実行する
        ed = encDecWrite(img, ch, args.quality)
        with chainer.using_config('train', False):
            out_imgs.append(
                predict(model, IMG.splitSQ(ed, size), args.batch, ed.shape, sr, args.gpu)
            )

    # 推論実行した各画像を結合してサイズを調整する
    img = stackImages(out_imgs, args.img_rate)
    # 生成結果の表示
    cv2.imshow('predict some snapshots', img)
    cv2.waitKey()
    # 生成結果の保存
    cv2.imwrite(F.getFilePath(args.out_path, 'snapshots.jpg'), img)


if __name__ == '__main__':
    main(command())

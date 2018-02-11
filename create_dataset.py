#!/usr/bin/env python3
# -*-coding: utf-8 -*-
#
help = '画像を読み込んでデータセットを作成する'
#

import os
import cv2
import argparse
import numpy as np

import Lib.imgfunc as IMG
import Tools.func as F


def command():
    parser = argparse.ArgumentParser(description=help)
    parser.add_argument('jpeg', nargs='+',
                        help='使用する画像のパス')
    parser.add_argument('--channel', '-c', type=int, default=1,
                        help='画像のチャンネル数 [default: 1 channel]')
    parser.add_argument('--img_size', '-s', type=int, default=32,
                        help='生成される画像サイズ [default: 32 pixel]')
    parser.add_argument('--round', '-r', type=int, default=1000,
                        help='切り捨てる数 [default: 1000]')
    parser.add_argument('--quality', '-q', type=int, default=5,
                        help='画像の圧縮率 [default: 5]')
    parser.add_argument('--train_per_all', '-t', type=float, default=0.9,
                        help='画像数に対する学習用画像の割合 [default: 0.9]')
    parser.add_argument('-o', '--out_path', default='./result/',
                        help='・ (default: ./result/)')
    return parser.parse_args()


def main(args):
    # OpenCV形式で画像を読み込むために
    # チャンネル数をOpenCVのフラグ形式に変換する
    ch = IMG.getCh(args.channel)
    # OpenCV形式で画像をリストで読み込む
    print('read images...')
    imgs = [cv2.imread(name, ch) for name in args.jpeg]
    # 画像を圧縮して分割する（学習の入力データに相当）
    print('split images...')
    x, _ = IMG.split(
        IMG.rotate(IMG.encodeDecode(imgs, ch, args.quality)),
        args.img_size,
        args.round
    )
    # 画像を分割する（正解データに相当）
    y, _ = IMG.split(
        IMG.rotate(imgs),
        args.img_size,
        args.round
    )

    # 画像の並び順をシャッフルするための配列を作成する
    # compとrawの対応を崩さないようにシャッフルしなければならない
    # また、train_sizeで端数を切り捨てる
    print('shuffle images...')
    shuffle = np.random.permutation(range(len(x)))
    train_size = int(len(x) * args.train_per_all)
    train_x = x[shuffle[:train_size]]
    train_y = y[shuffle[:train_size]]
    test_x = x[shuffle[train_size:]]
    test_y = y[shuffle[train_size:]]
    print('train x/y:{0}/{1}'.format(train_x.shape, train_y.shape))
    print('test  x/y:{0}/{1}'.format(test_x.shape, test_y.shape))

    # 生成したデータをnpz形式でデータセットとして保存する
    # ここで作成したデータの中身を確認する場合はnpz2jpg.pyを使用するとよい
    print('save npz...')
    size_str = '_' + str(args.img_size).zfill(2) + 'x' + \
        str(args.img_size).zfill(2)
    num_str = '_' + str(train_x.shape[0]).zfill(6)
    np.savez(F.getFilePath(args.out_path, 'train' + size_str + num_str),
             x=train_x, y=train_y)
    num_str = '_' + str(test_x.shape[0]).zfill(6)
    np.savez(F.getFilePath(args.out_path, 'test' + size_str + num_str),
             x=test_x, y=test_y)


if __name__ == '__main__':
    args = command()
    F.argsPrint(args)
    main(args)

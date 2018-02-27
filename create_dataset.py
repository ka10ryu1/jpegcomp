#!/usr/bin/env python3
# -*-coding: utf-8 -*-
#
help = '画像を読み込んでデータセットを作成する'
#

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


def saveNPZ(x, y, name, folder, size):
    """
    入力データと正解データをNPZ形式で保存する
    [in] x:      保存する入力データ
    [in] y:      保存する正解データ
    [in] name:   保存する名前
    [in] folder: 保存するフォルダ
    [in] size:   データ（正方形画像）のサイズ
    """

    size_str = '_' + str(size).zfill(2) + 'x' + str(size).zfill(2)
    num_str = '_' + str(x.shape[0]).zfill(6)
    np.savez(F.getFilePath(folder, name + size_str + num_str), x=x, y=y)


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
    train_x = IMG.imgs2arr(x[shuffle[:train_size]], dtype=np.float16)
    train_y = IMG.imgs2arr(y[shuffle[:train_size]], dtype=np.float16)
    test_x = IMG.imgs2arr(x[shuffle[train_size:]], dtype=np.float16)
    test_y = IMG.imgs2arr(y[shuffle[train_size:]], dtype=np.float16)
    print('train x/y:{0}/{1}'.format(train_x.shape, train_y.shape))
    print('test  x/y:{0}/{1}'.format(test_x.shape, test_y.shape))

    # 生成したデータをnpz形式でデータセットとして保存する
    # ここで作成したデータの中身を確認する場合はnpz2jpg.pyを使用するとよい
    print('save npz...')
    saveNPZ(train_x, train_y, 'train', args.out_path, args.img_size)
    saveNPZ(test_x, test_y, 'test', args.out_path, args.img_size)


if __name__ == '__main__':
    args = command()
    F.argsPrint(args)
    main(args)

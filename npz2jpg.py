#!/usr/bin/env python3
# -*-coding: utf-8 -*-
#
help = '作成したデータセット（.npz）の中身を画像として出力する'
#

import os
import cv2
import sys
import argparse
import numpy as np

[sys.path.append(d) for d in ['./Lib/', '../Lib/'] if os.path.isdir(d)]
from func import argsPrint, getFilePath
import imgfunc as IMG


def command():
    parser = argparse.ArgumentParser(description=help)
    parser.add_argument('npz',
                        help='使用するnpzファイルのパス')
    parser.add_argument('--img_num', '-n', type=int, default=10,
                        help='読み込む画像数 [default: 10]')
    parser.add_argument('--random_seed', '-s', type=int, default=2,
                        help='乱数シード [default: 2, random: -1]')
    parser.add_argument('--img_rate', '-r', type=float, default=1,
                        help='画像サイズの倍率 [default: 1]')
    parser.add_argument('-o', '--out_path', default='./result/',
                        help='画像の出力先 [default: ./result/]')
    parser.add_argument('--only_x', action='store_true',
                        help='xだけを表示する')
    return parser.parse_args()


def main(args):
    # NPZ形式のファイルを読み込む
    np_arr = np.load(args.npz)
    x = IMG.arr2imgs(np_arr['x'])
    if not args.only_x:
        y = IMG.arr2imgs(np_arr['y'])
        ch = 3
        if(x.shape[ch] > y.shape[ch]):
            y = [cv2.cvtColor(i, cv2.COLOR_GRAY2RGB) for i in y]
            y = np.array(y)

        if(x.shape[ch] > y.shape[ch]):
            x = [cv2.cvtColor(i, cv2.COLOR_GRAY2RGB) for i in x]
            x = np.array(x)

    # 全てを画像化するのは無駄なのでランダムに抽出する
    if(args.random_seed >= 0):
        np.random.seed(args.random_seed)

    shuffle = np.random.permutation(range(len(x)))
    # ランダムに抽出した画像を結合する
    # 上半分にはxの画像をimg_numの数だけ
    # 下半分にはyの画像をimg_numの数だけ結合した画像を作成する
    if not args.only_x:
        img = np.vstack((np.hstack(x[shuffle[:args.img_num]]),
                         np.hstack(y[shuffle[:args.img_num]])))
    else:
        img = np.hstack(x[shuffle[:args.img_num]])

    # そのままのサイズでは画像が小さいので、拡大する
    img = IMG.resize(img, args.img_rate, cv2.INTER_NEAREST)
    # 作成した画像を表示・保存する
    cv2.imshow('test', img)
    cv2.waitKey(0)
    cv2.imwrite(getFilePath(args.out_path, 'npz2jpg', '.jpg'), img)


if __name__ == '__main__':
    args = command()
    argsPrint(args)
    main(args)

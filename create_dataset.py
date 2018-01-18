#!/usr/bin/env python3
# -*-coding: utf-8 -*-
#
help = '画像を読み込んでデータセットを作成する'
#

import os
import cv2
import argparse
import numpy as np

from func import argsPrint, getCh, imgEncodeDecode, imgSplit


def command():
    parser = argparse.ArgumentParser(description=help)
    parser.add_argument('jpeg', nargs='+',
                        help='使用する画像のパス')
    parser.add_argument('--channel', '-c', type=int, default=1,
                        help='画像のチャンネル数（default: 1 channel）')
    parser.add_argument('--img_size', '-s', type=int, default=32,
                        help='生成される画像サイズ（default: 32 pixel）')
    parser.add_argument('--round', '-r', type=int, default=1000,
                        help='切り捨てる数（default: 1000）')
    parser.add_argument('--quality', '-q', type=int, default=5,
                        help='画像の圧縮率（default: 5）')
    parser.add_argument('--train_per_all', '-t', type=float, default=0.85,
                        help='画像数に対する学習用画像の割合（default: 0.85）')
    parser.add_argument('-o', '--out_path', default='./result/',
                        help='・ (default: ./result/)')
    return parser.parse_args()


def main(args):
    # OpenCV形式で画像を読み込むために
    # チャンネル数をOpenCVのフラグ形式に変換する
    ch = getCh(args.channel)
    # OpenCV形式で画像をリストで読み込む
    imgs = [cv2.imread(name, ch) for name in args.jpeg]
    # 画像を圧縮して分割する（学習の入力データに相当）
    comp, _ = imgSplit(imgEncodeDecode(imgs, ch, args.quality),
                       args.img_size, args.round)
    # 画像を分割する（正解データに相当）
    raw, _ = imgSplit(imgs, args.img_size, args.round)

    # 画像の並び順をシャッフルするための配列を作成する
    # compとrawの対応を崩さないようにシャッフルしなければならない
    # また、train_sizeで端数を切り捨てる
    shuffle = np.random.permutation(range(len(comp)))
    train_size = int(len(comp) * args.train_per_all)
    train_comp = comp[shuffle[:train_size]]
    train_raw = raw[shuffle[:train_size]]
    test_comp = comp[shuffle[train_size:]]
    test_raw = raw[shuffle[train_size:]]
    print('train comp:', train_comp.shape)
    print('      raw: ', train_raw.shape)
    print('test comp: ', test_comp.shape)
    print('     raw:  ', test_raw.shape)

    # 保存するフォルダがなければ生成する
    if not os.path.isdir(args.out_path):
        os.makedirs(args.out_path)

    # 生成したデータをnpz形式でデータセットとして保存する
    # ここで作成したデータの中身を確認する場合はnpz2jpg.pyを使用するとよい
    np.savez(os.path.join(args.out_path, 'train'),
             comp=train_comp, raw=train_raw)
    np.savez(os.path.join(args.out_path, 'test'),
             comp=test_comp, raw=test_raw)


if __name__ == '__main__':
    args = command()
    argsPrint(args)
    main(args)

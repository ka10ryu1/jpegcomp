#!/usr/bin/env python3
# -*-coding: utf-8 -*-
#
help = '3枚の画像を連結する（org, comp, restration）'
#

import os
import sys
import cv2
import argparse
import numpy as np

[sys.path.append(d) for d in ['./Tools/', '../Tools/'] if os.path.isdir(d)]
from func import argsPrint, getFilePath
[sys.path.append(d) for d in ['./Lib/', '../Lib/'] if os.path.isdir(d)]
import imgfunc as IMG


def command():
    parser = argparse.ArgumentParser(description=help)
    parser.add_argument('image', nargs='+',
                        help='入力データセットのフォルダ (default: ./result/)')
    parser.add_argument('-o', '--offset', type=int, default=50,
                        help='入力データセットのフォルダ (default: ./result/)')
    parser.add_argument('-s', '--img_width', type=int, default=333,
                        help='入力データセットのフォルダ (default: ./result/)')
    parser.add_argument('-r', '--img_rate', type=float, default=1,
                        help='画像サイズの倍率（default: 1）')
    parser.add_argument('--channel', '-c', type=int, default=1,
                        help='画像のチャンネル数（default: 1 channel）')
    parser.add_argument('-op', '--out_path', default='./result/',
                        help='・ (default: ./result/)')
    return parser.parse_args()


def titleInsert(img, text, header_size,
                color=(255, 255, 255), org=(10, 20), scale=0.5, thick=1):
    """
    画像の上部にテキストを結合する
    [in] img:         結合する画像
    [in] text:        結合するテキスト
    [in] header_size: テキストを書き込む場所のサイズ
    [in] color:       テキストを書き込む場所の色
    [in] org:         テキストを書き込む位置
    [in] scale:       テキストのスケール
    [in] thick:       テキストの太さ
    [out] テキストが上部に結合された画像
    """

    if(len(img.shape) == 2):
        header = np.zeros(header_size[:2], dtype=np.uint8)
    else:
        header = np.zeros(header_size, dtype=np.uint8)

    img = np.vstack([header, img])
    return cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                       scale, color, thick, cv2.LINE_AA)


def stackImages(imgs, thick=1, color=(0, 0, 0), flg=cv2.BORDER_CONSTANT):
    """
    画像を横に連結する
    [in] imgs:  連結する画像のリスト
    [in] thick: 画像を区切る線の太さ
    [in] color: 画像を区切る線の色
    [in] flg:   境界線のフラグ
    [out] 連結された画像
    """

    imgs = [cv2.copyMakeBorder(img, 0, thick, 0, thick, flg, value=color)
            for img in imgs]
    return np.hstack(imgs)


def concat3Images(imgs, start_pos, img_width, ch, rate,
                  text=['[Original]', '[Compression]', '[Restoration]']):
    """
    3枚の画像を任意の部分切り抜き、その上部にテキストを追加し、連結する
    [in] imgs:      連結する画像のリスト
    [in] start_pos: 画像を切り抜く開始ピクセル
    [in] img_width: 画像を切り抜く幅ピクセル
    [in] ch:        画像のチャンネル数
    [in] rate:      画像の縮尺
    [in] text:      画像に挿入するテキスト
    [out] 連結された画像
    """

    height = np.min([i.shape[0] for i in imgs])
    end_pos = start_pos + img_width
    if(ch == 1):
        imgs = [i[:height, start_pos:end_pos] for i in imgs]
    else:
        imgs = [i[:height, start_pos:end_pos, :] for i in imgs]

    imgs = [IMG.resize(i, rate) for i in imgs]
    header_size = (30, int(img_width * rate), 3)
    imgs = [titleInsert(i, t, header_size) for i, t in zip(imgs, text)]
    return stackImages(imgs, thick=1, color=(0, 0, 0))


def main(args):
    ch = IMG.getCh(args.channel)
    imgs = [cv2.imread(name, ch) for name in args.image]
    #text = ['[hitotsume]', '[futatsume]', '[mittsume]']
    img = concat3Images(
        imgs, args.offset, args.img_width, args.channel, args.img_rate
    )

    cv2.imshow('test', img)
    cv2.waitKey()
    cv2.imwrite(getFilePath(args.out_path, 'concat', '.jpg'), img)


if __name__ == '__main__':
    args = command()
    argsPrint(args)
    main(args)

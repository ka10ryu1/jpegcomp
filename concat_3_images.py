#!/usr/bin/env python3
# -*-coding: utf-8 -*-
#
help = '3枚の画像を連結する（org, comp, restration）'
#

import cv2
import argparse
import numpy as np

from Tools.func import argsPrint, getFilePath
import Lib.imgfunc as IMG


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


def getSize(img, rate):
    return (int(img.shape[1] * rate), int(img.shape[0] * rate))


def titleInsert(img, text, header_size,
                color=(255, 255, 255), org=(10, 20), scale=0.5, thick=1):
    if(len(img.shape) == 2):
        header = np.zeros(header_size[:2], dtype=np.uint8)
    else:
        header = np.zeros(header_size, dtype=np.uint8)

    img = np.vstack([header, img])
    return cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                       scale, color, thick, cv2.LINE_AA)


def concatImage(imgs, thick=1, color=(0, 0, 0)):
    flg = cv2.BORDER_CONSTANT
    val = color
    imgs = [cv2.copyMakeBorder(img, 0, thick, 0, thick, flg, value=val)
            for img in imgs]
    return np.hstack(imgs)


def main(args):
    ch = IMG.getCh(args.channel)
    imgs = [cv2.imread(name, ch) for name in args.image]
    height = np.min([i.shape[0] for i in imgs])
    start_pos = args.offset
    end_pos = start_pos + args.img_width
    if(args.channel == 1):
        imgs = [i[:height, start_pos:end_pos] for i in imgs]
    else:
        imgs = [i[:height, start_pos:end_pos, :] for i in imgs]

    imgs = [cv2.resize(i, getSize(i, args.img_rate),
                       cv2.INTER_NEAREST) for i in imgs]
    text = ['[Original]', '[Compression]', '[Restoration]']
    #text = ['[hitotsume]', '[futatsume]', '[mittsume]']
    header_size = (30, int(args.img_width * args.img_rate), 3)
    imgs = [titleInsert(i, t, header_size) for i, t in zip(imgs, text)]
    img = concatImage(imgs, thick=1, color=(0, 0, 0))

    cv2.imshow('test', img)
    cv2.waitKey()
    cv2.imwrite(getFilePath(args.out_path, 'concat', '.jpg'), img)


if __name__ == '__main__':
    args = command()
    argsPrint(args)
    main(args)

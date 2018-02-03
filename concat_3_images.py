#!/usr/bin/env python3
# -*-coding: utf-8 -*-
#
help = ''
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
    parser.add_argument('-s', '--img_width', type=int, default=333,
                        help='入力データセットのフォルダ (default: ./result/)')
    parser.add_argument('--channel', '-c', type=int, default=1,
                        help='画像のチャンネル数（default: 1 channel）')
    parser.add_argument('-o', '--out_path', default='./result/',
                        help='・ (default: ./result/)')
    return parser.parse_args()


def main(args):
    ch = IMG.getCh(args.channel)
    imgs = [cv2.imread(name, ch) for name in args.image]
    imgs = [i[:, :args.img_width] for i in imgs]

    text = ['[Original]', '[Compression]', '[Restoration]']
    black = (0, 0, 0)
    white = (255, 255, 255)
    org = (10, 20)
    scale = 0.5
    thick = 1

    header = np.zeros((30, args.img_width), dtype=np.uint8)
    imgs = [np.vstack([header, i]) for i in imgs]

    # img, text, org, fontFace, fontScale, color, thickness, lineType
    [cv2.putText(i, t, org, cv2.FONT_HERSHEY_SIMPLEX, scale, white,
                 thick, cv2.LINE_AA) for i, t in zip(imgs, text)]

    imgs = [cv2.copyMakeBorder(img, 0, 1, 0, 1, cv2.BORDER_CONSTANT, value=black)
            for img in imgs]

    img = np.hstack(imgs)
    cv2.imshow('test', img)
    cv2.waitKey()
    cv2.imwrite(getFilePath(args.out_path, 'concat', '.jpg'), img)


if __name__ == '__main__':
    args = command()
    argsPrint(args)
    main(args)

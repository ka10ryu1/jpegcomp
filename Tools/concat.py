#!/usr/bin/env python3
# -*-coding: utf-8 -*-
#
help = '複数の画像を任意の行列で結合する'
#

import os
import sys
import cv2
import argparse
import numpy as np

[sys.path.append(d) for d in ['./Tools/', '../Tools/'] if os.path.isdir(d)]
import func as F


def command():
    parser = argparse.ArgumentParser(description=help)
    parser.add_argument('jpeg', nargs='+',
                        help='使用する画像のパス')
    parser.add_argument('--out_path', '-o', default='./result/',
                        help='生成物の保存先 [default: ./result/]')
    parser.add_argument('--row', '-r', type=int, default=10,
                        help='画像を連結する行 [default: 10]')
    parser.add_argument('--line_width', '-lw', type=int, default=2,
                        help='画像を連結する行 [default: 2]')
    parser.add_argument('--resize', '-rs', type=float, default=0.5,
                        help='画像の縮尺 [default: 0.5]')
    return parser.parse_args()


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


def main(args):
    imgs = [cv2.imread(name) for name in args.jpeg]
    h = np.max([img.shape[0] for img in imgs])
    flg = cv2.BORDER_REFLECT_101
    imgs = [
        cv2.copyMakeBorder(img, 0, h - img.shape[0], 0, 0, flg)
        for img in imgs
    ]
    #[print(img.shape) for img in imgs]

    lw = args.line_width
    flg = cv2.BORDER_CONSTANT
    imgs = [
        cv2.copyMakeBorder(img, 0, lw, 0, lw, flg, value=(0, 0, 0))
        for img in imgs
    ]

    size = np.arange(len(imgs)).reshape(-1, args.row).shape
    buf = [np.vstack(imgs[i * size[0]: (i + 1) * size[0]])
           for i in range(size[1])]
    img = np.hstack(buf)

    resize = (int(img.shape[1] * args.resize),
              int(img.shape[0] * args.resize))
    img = cv2.resize(img, resize, cv2.INTER_NEAREST)

    name = F.getFilePath(args.out_path, 'concat', '.jpg')
    print('save:', name)
    cv2.imwrite(name, img)


if __name__ == '__main__':
    args = command()
    F.argsPrint(args)
    main(args)

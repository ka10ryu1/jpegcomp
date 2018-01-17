#!/usr/bin/env python3
help = '画像を読み込んでデータセットを作成する'

import cv2
import argparse
import numpy as np

from func import argsPrint, getFilePath


def command():
    parser = argparse.ArgumentParser(description=help)
    parser.add_argument('npz',
                        help='使用するnpzファイルのパス')
    parser.add_argument('--img_num', '-n', type=int, default=10,
                        help='読み込む画像数（default: 10）')
    parser.add_argument('--random_seed', '-s', type=int, default=2,
                        help='乱数シード（default: 2）')
    parser.add_argument('--img_rate', '-r', type=int, default=4,
                        help='画像サイズの倍率（default: 4）')
    parser.add_argument('-o', '--out_path', default='./result/',
                        help='・ (default: ./result/)')
    return parser.parse_args()


def main(args):
    np_arr = np.load(args.npz)
    comp = np_arr['comp']
    raw = np_arr['raw']

    np.random.seed(args.random_seed)
    shuffle = np.random.permutation(range(len(comp)))
    img = np.vstack((np.hstack(comp[shuffle[:args.img_num]]),
                     np.hstack(raw[shuffle[:args.img_num]])))
    w, h = img.shape[:2]
    img = cv2.resize(img, (h * args.img_rate, w * args.img_rate),
                     cv2.INTER_NEAREST)
    cv2.imshow('test', img)
    cv2.waitKey(0)
    cv2.imwrite(getFilePath(args.out_path, 'npz2jpg', '.jpg'), img)


if __name__ == '__main__':
    args = command()
    argsPrint(args)
    main(args)

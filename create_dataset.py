#!/usr/bin/env python3
help = '画像を読み込んでデータセットを作成する'

import cv2
import argparse
import numpy as np


def command():
    parser = argparse.ArgumentParser(description=help)
    parser.add_argument('jpeg', nargs='+',
                        help='使用する画像のパス')
    parser.add_argument('--img_size', '-s', type=int, default=32,
                        help='生成される画像サイズ（default: 32 pixel）')
    parser.add_argument('--train_per_all', '-t', type=float, default=0.85,
                        help='画像数に対する学習用画像の割合ズ（default: 0.85）')
    return parser.parse_args()


def imgEncodeDecode(in_imgs, quality=5):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    out_imgs = []

    for img in in_imgs:
        result, encimg = cv2.imencode('.jpg', img, encode_param)
        if False == result:
            print('could not encode image!')
            exit()

        decimg = cv2.imdecode(encimg, 1)
        out_imgs.append(decimg)

    return out_imgs


def imgSplit(imgs, size):
    v_size = imgs[0].shape[0] // size * size
    h_size = imgs[0].shape[1] // size * size
    imgs = [i[:v_size, :h_size] for i in imgs]
    v_split = imgs[0].shape[0] // size
    h_split = imgs[0].shape[1] // size
    out_imgs = []
    [[out_imgs.extend(np.vsplit(h_img, v_split))
      for h_img in np.hsplit(img, h_split)] for img in imgs]
    return np.array(out_imgs)


def main(args):
    print(args.jpeg)
    imgs = [cv2.imread(name) for name in args.jpeg]

    comp = imgSplit(imgEncodeDecode(imgs), args.img_size)
    raw = imgSplit(imgs, args.img_size)

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

    np.savez('train', comp=train_comp, raw=train_raw)
    np.savez('test', comp=test_comp, raw=test_raw)


if __name__ == '__main__':
    args = command()
    main(args)

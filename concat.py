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
import imgfunc as IMG


def command():
    parser = argparse.ArgumentParser(description=help)
    parser.add_argument('jpeg', nargs='+',
                        help='使用する画像のパス')
    parser.add_argument('--out_path', '-o', default='./result/',
                        help='生成物の保存先 [default: ./result/]')
    parser.add_argument('--row', '-r', type=int, default=-1,
                        help='画像を連結する行（負数で自動計算） [default: -1]')
    parser.add_argument('--line_width', '-lw', type=int, default=2,
                        help='画像を連結する行 [default: 2]')
    parser.add_argument('--resize', '-rs', type=float, default=0.5,
                        help='画像の縮尺 [default: 0.5]')
    return parser.parse_args()


def makeDivisorList(num):
    """
    入力された数の約数に1を加えたリストを返す
    [in]  num:          約数を計算したい数
    [out] divisor_list: numの約数に1を加えたリスト
    """

    if num < 1:
        return [0]
    elif num == 1:
        return [1]
    else:
        divisor_list = [i for i in range(2, num // 2 + 1) if num % i == 0]
        divisor_list.append(1)

        return divisor_list


def stackImgAndShape(imgs, row):
    """
    画像を縦横に連結するための画像リストと縦横画像数情報を取得する
    [in]  imgs: 連結したい入力画像リスト
    [in]  row:
    [out] 画像リスト
    [out] 縦横画像数情報
    """

    # row=0は強制的に1にする
    if row == 0:
        row = 1

    # 入力画像リストがrowで割り切れない時用に
    # 黒塗画像を用意するが、3枚の根拠はない
    if row > 3 or 0 > row:
        bk = np.zeros(imgs[0].shape, dtype=np.uint8)
        imgs.append(bk)
        imgs.append(bk)
        imgs.append(bk)

    # rowが負数の場合はrowを自動計算する
    if 0 > row:
        # 黒塗画像を0枚含んだ状態でdiv_listが3以上になればdivとimgsを決定
        # div_listが十分でなかった場合、
        # 黒塗画像を1枚含んだ状態でdiv_listが3以上になればdivとimgsを決定
        # これを黒塗画像3枚まで続ける
        for i in range(3, 0, -1):
            # 画像の数で約数を探す
            div_list = makeDivisorList(len(imgs[:-i]))
            if(len(div_list) > 2):
                # rowは約数のリストの中心の値を取得する
                # これにより正方形に近い連結画像が生成できる
                row = div_list[len(div_list) // 2]
                imgs = imgs[:-i]
                break

    else:
        img_len = len(imgs) // row * row
        imgs = imgs[:img_len]

    return np.array(imgs), np.arange(len(imgs)).reshape(-1, row)


def makeBorder(img, top, bottom, left, right, flg, value=None):
    """
    cv2.copyMakeBorder()のラッパー関数なので詳細は省く
    """

    if flg == cv2.BORDER_CONSTANT:
        return cv2.copyMakeBorder(img, top, bottom, left, right, flg, value=value)
    else:
        return cv2.copyMakeBorder(img, top, bottom, left, right, flg)


def main(args):
    # 画像を読み込む
    imgs = [cv2.imread(name) for name in args.jpeg if IMG.isImgPath(name)]
    # concatするためにすべての画像の高さを統一する
    h = np.max([img.shape[0] for img in imgs])
    imgs = [IMG.resize(img, h / img.shape[0]) for img in imgs]
    # concatするためにすべての画像の幅を統一する
    flg = cv2.BORDER_REFLECT_101
    w = np.max([img.shape[1] for img in imgs])
    imgs = [makeBorder(img, 0, 0, 0, w - img.shape[1], flg) for img in imgs]
    # 画像に黒縁を追加する
    flg = cv2.BORDER_CONSTANT
    lw = args.line_width
    imgs = [makeBorder(img, 0, lw, 0, lw, flg, (0, 0, 0)) for img in imgs]
    # 縦横に連結するための画像リストと縦横情報を取得する
    imgs, size = stackImgAndShape(imgs, args.row)
    # 画像を連結してリサイズする
    buf = [np.vstack(imgs[s]) for s in size]
    img = IMG.resize(np.hstack(buf), args.resize)
    # 連結された画像を保存する
    name = F.getFilePath(args.out_path, 'concat', '.jpg')
    print('save:', name)
    cv2.imwrite(name, img)


if __name__ == '__main__':
    args = command()
    F.argsPrint(args)
    main(args)

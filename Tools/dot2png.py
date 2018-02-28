#!/usr/bin/env python3
# -*-coding: utf-8 -*-
#
help = 'dot言語で記述されたファイルをPNG形式に変換する'
#

import os
# pydot: 1.2.4
# graphviz: 0.8.2
import pydot
import argparse

from func import argsPrint, getFilePath


def command():
    parser = argparse.ArgumentParser(description=help)
    parser.add_argument('dot', nargs='+',
                        help='使用するdotファイルのパス')
    parser.add_argument('-e', '--ext', default='png',
                        help='保存する拡張子 (default: png, other: pdf, svg)')
    parser.add_argument('-o', '--out_path', default='./result/',
                        help='出力するフォルダ (default: ./result/)')
    return parser.parse_args()


def main(args):
    for name in args.dot:
        # dot言語で記述されたファイルを読み込む
        (graph,) = pydot.graph_from_dot_file(name)
        # 保存用の名前を抽出する
        name, _ = os.path.splitext(os.path.basename(name))
        # 形式を選択して保存する
        if(args.ext == 'png'):
            graph.write_png(getFilePath(args.out_path, name, '.png'))
        elif(args.ext == 'pdf'):
            graph.write_pdf(getFilePath(args.out_path, name, '.pdf'))
        elif(args.ext == 'svg'):
            graph.write_svg(getFilePath(args.out_path, name, '.svg'))
        else:
            print('[ERROR] ext option miss:', args.ext)


if __name__ == '__main__':
    args = command()
    argsPrint(args)
    main(args)

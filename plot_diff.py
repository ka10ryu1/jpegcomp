#!/usr/bin/env python3
# -*-coding: utf-8 -*-
#
help = 'logファイルの複数比較'
#

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

from func import argsPrint, getFilePath, sortTimeStamp


def command():
    parser = argparse.ArgumentParser(description=help)
    parser.add_argument('log_dir', nargs='+',
                        help='入力データセットのフォルダ')
    parser.add_argument('--auto_ylim', action='store_true',
                        help='ylim自動設定')
    parser.add_argument('-l', '--label', default='loss',
                        help='取得するラベル(default: loss, other: lr, all)')
    parser.add_argument('-o', '--out_path', default='./result/',
                        help='生成物の保存先(default: ./result/)')
    parser.add_argument('--no_show', action='store_true',
                        help='plt.show()を使用しない')

    return parser.parse_args()


def jsonRead(path):
    """
    chainerのextensionで出力されたlogをjsonで読み込む
    [in]  path: logのパス
    [out] d:    読み込んだ辞書データ
    """

    try:
        with open(path, 'r') as f:
            d = json.load(f)

    except json.JSONDecodeError as e:
        print('JSONDecodeError: ', e)
        exit()

    return d


def subplot(sub, val, log, ylim, line, header):
    """
    subplotを自動化
    [in] sub:    subplotオブジェクト
    [in] val:    入力する値のリスト
    [in] log:    入力するラベルのリスト
    [in] ylim:   auto_ylimを使用する場合はTrue
    [in] header: ラベルのヘッダ
    """

    # グリッドを灰色の点線で描画する
    sub.grid(which='major', color='gray', linestyle=':')
    sub.grid(which='minor', color='gray', linestyle=':')
    sub.set_yscale("log")
    # args.auto_ylimが設定された場合、ylimを設定する
    # ymax: 各データの1/8番目（400個データがあれば50番目）のうち最小の数を最大値とする
    # ymin: 各データのうち最小の数X0.98を最小値とする
    if ylim:
        ymax = np.min([i[int(len(i) / 8)] for i in val])
        ymin = np.min([np.min(i)for i in val]) * 0.98
        sub.set_ylim([ymin, ymax])
        print('ymin:{0:.4f}, ymax:{1:.4f}'.format(ymin, ymax))

    # プロット
    def getX(y):
        return list(range(1, len(y)+1))

    def getY(y):
        return np.array(y)

    def getLabel(header, body):
        return '[' + header + '] ' + body

    [sub.plot(getX(v), getY(v), label=getLabel(header, d), linestyle=line)
     for v, d in zip(val, log)]


def savePNG(plt, loc, name, dpi=200):
    """
    png形式での保存を自動化
    [in] plt:  pltオブジェクト
    [in] loc:  ラベルの位置
    [in] name: 保存するファイル名
    [in] dpi:  保存時の解像度
    """

    plt.legend(loc=loc)
    plt.savefig(getFilePath(args.out_path, name, '.png'), dpi=dpi)


def plot(args, loc, name, solid_line, dotted_line='', no_show=False):
    """
    プロットメイン部
    [in] args:   オプション引数
    [in] loc:    ラベルの位置
    [in] name:   保存するファイル名
    [in] solid_line: 探索ラベル（実線）
    [in] dotted_line: 探索ラベル（点線）
    """

    sol = []
    dot = []
    log_file = []
    for l in sortTimeStamp(args.log_dir, '.log'):
        log_file.append(l)
        print(log_file[-1])
        data = jsonRead(log_file[-1])
        sol.append([i[solid_line] for i in data if(solid_line in i.keys())])
        dot.append([i[dotted_line] for i in data if(dotted_line in i.keys())])

    # logファイルが見つからなかった場合、ここで終了
    if not sol:
        print('[Error] .log not found')
        exit()

    if len(sol[0]) == 0:
        print('[Error] data not found:', solid_line)
        return 0

    # 対数グラフの設定
    f = plt.figure(figsize=(10, 6))
    a = f.add_subplot(111)
    plt.xlabel('epoch')
    plt.ylabel(name.split('_')[-1])
    subplot(a, sol, log_file, args.auto_ylim, '-', 'test ')
    plt.gca().set_prop_cycle(None)
    subplot(a, dot, log_file, args.auto_ylim, ':', 'train')

    # グラフの保存と表示
    savePNG(plt, loc, name)
    if not no_show:
        plt.show()


def main(args):
    if(args.label == 'loss' or args.label == 'all'):
        plot(args, 'upper right', 'plot_diff_loss',
             'validation/main/loss', 'main/loss',
             no_show=args.no_show)

    if(args.label == 'acc' or args.label == 'all'):
        plot(args, 'lower right', 'plot_diff_acc',
             'validation/main/accuracy', 'main/accuracy',
             no_show=args.no_show)

    if(args.label == 'lr' or args.label == 'all'):
        plot(args, 'lower right', 'plot_diff_lr', 'lr',
             no_show=args.no_show)


if __name__ == '__main__':
    args = command()
    argsPrint(args)
    main(args)

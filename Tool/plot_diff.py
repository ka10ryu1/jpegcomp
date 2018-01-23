#!/usr/bin/env python3
# -*-coding: utf-8 -*-
#
help = 'logファイルの複数比較'
#

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

[sys.path.append(d) for d in ['./Lib/', '../Lib/'] if os.path.isdir(d)]
from myfunc import argsPrint, getFilePath


def command():
    parser = argparse.ArgumentParser(description=help)
    parser.add_argument('log_dir', nargs='+',
                        help='入力データセットのフォルダ')
    parser.add_argument('--auto_ylim', action='store_true',
                        help='ylim自動設定')
    parser.add_argument('-o', '--out_path', default='./result/',
                        help='生成物の保存先(default: ./result/)')

    return parser.parse_args()


def jsonRead(path):
    try:
        with open(path, 'r') as f:
            d = json.load(f)

    except json.JSONDecodeError as e:
        print('JSONDecodeError: ', e)
        exit()

    return d


def main(args):
    vml = []
    for d in args.log_dir:
        if not os.path.isdir(d):
            print('[Error] this is not dir:', d)
            continue

        for l in os.listdir(d):
            name, ext = os.path.splitext(os.path.basename(l))
            if(ext == '.log'):
                print(l)
                data = jsonRead(os.path.join(d, l))
                buf = [i['validation/main/loss'] for i in data]
                vml.append(buf)

    if len(vml) == 0:
        print('[Error] .log not found')
        exit()

    f = plt.figure()
    a = f.add_subplot(111)
    a.grid(which='major', color='black', linestyle='-')
    a.grid(which='minor', color='black', linestyle='-')
    plt.yscale("log")

    [a.plot(np.array(v), label=d) for v, d in zip(vml, args.log_dir)]
    if args.auto_ylim:
        ymax = np.min([i[int(len(i) / 8)] for i in vml])
        ymin = np.min([np.min(i)for i in vml]) * 0.98
        plt.ylim([ymin, ymax])
        print('ymin:{0:.4f}, ymax:{1:.4f}'.format(ymin, ymax))

    plt.legend()
    plt.savefig(getFilePath(args.out_path, 'plot_diff', '.png'), dpi=200)
    plt.show()


if __name__ == '__main__':
    args = command()
    argsPrint(args)
    main(args)

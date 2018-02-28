#!/usr/bin/env python3
# -*-coding: utf-8 -*-
#
help = '便利機能'
#

import os
import inspect
from pathlib import Path


def argsPrint(p, bar=30):
    """
    argparseの parse_args() で生成されたオブジェクトを入力すると、
    integersとaccumulateを自動で取得して表示する
    [in] p: parse_args()で生成されたオブジェクト
    [in] bar: 区切りのハイフンの数
    """

    print('-' * bar)
    args = [(i, getattr(p, i)) for i in dir(p) if not '_' in i[0]]
    for i, j in args:
        if isinstance(j, list):
            print('{0}[{1}]:'.format(i, len(j)))
            [print('\t{}'.format(k)) for k in j]
        else:
            print('{0}:\t{1}'.format(i, j))

    print('-' * bar)


def getFilePath(folder, name, ext=''):
    if not os.path.isdir(folder):
        os.makedirs(folder)

    return os.path.join(folder, name + ext)


def sortTimeStamp(folder_list, ext):
    path = []
    [path.extend(list(Path(f).glob('*'+ext))) for f in folder_list]
    return sorted([x.as_posix() for x in path], key=os.path.getmtime)


def fileFuncLine():
    """
    この関数を呼び出すと、呼び出し先のファイル名、関数名、実行行数を取得できる
    デバッグ時に便利
    """

    funcname = inspect.currentframe().f_back.f_code.co_name
    filename = os.path.basename(
        inspect.currentframe().f_back.f_code.co_filename
    )
    lineno = inspect.currentframe().f_back.f_lineno
    return '>>> {0}, {1}(), {2}[line] <<<\n'.format(filename, funcname, lineno)

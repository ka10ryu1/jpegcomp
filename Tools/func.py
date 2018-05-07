#!/usr/bin/env python3
# -*-coding: utf-8 -*-
#
help = '便利機能'
#

import os
import inspect
from pathlib import Path
from watchdog.events import FileSystemEventHandler


class ChangeHandler(FileSystemEventHandler):

    def __init__(self):
        pass

    def on_created(self, event):
        filepath = event.src_path
        filename = os.path.basename(filepath)
        _, ext = os.path.splitext(filename)
        return filepath, filename, ext

    def on_modified(self, event):
        filepath = event.src_path
        filename = os.path.basename(filepath)
        _, ext = os.path.splitext(filename)
        return filepath, filename, ext

    def on_deleted(self, event):
        filepath = event.src_path
        filename = os.path.basename(filepath)
        _, ext = os.path.splitext(filename)
        return filepath, filename, ext


def argsPrint(p, bar=30):
    """
    argparseのparse_args() で生成されたオブジェクトを入力すると、
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


def args2dict(args):
    """
    argparseのparse_args() で生成されたオブジェクトを取得して辞書型に変換する
    [in]  parse_args() で生成されたオブジェクト
    [out] inの辞書型に変換した結果
    """
    return {i: getattr(args, i) for i in dir(args) if not '_' in i[0]}


def dict2json(folder, name, mydict, indent=4, sort_keys=True):
    """
    辞書型のオブジェクトをjson形式で保存する
    [in]  folder:    保存するフォルダ
    [in]  name:      保存するファイル名
    [in]  mydict:    保存したい辞書型のオブジェクト
    [in]  indent:    jsonで保存する際のインデント用のスペースの数
    [in]  sort_keys: jsonで保存する際の辞書をソートするフラグ
    """

    import json
    path = getFilePath(folder, name, '.json')
    with open(path, 'w') as f:
        json.dump(mydict, f, indent=4, sort_keys=True)


def checkModelType(path):
    """
    入力されたパスが.modelか.snapshotかそれ以外か判定し、
    load_npzのpathを設定する
    [in]  path:      入力されたパス
    [out] load_path: load_npzのpath
    """

    # 拡張子を正とする
    name, ext = os.path.splitext(os.path.basename(path))
    load_path = ''
    if(ext == '.model'):
        print('model read:', path)
    elif(ext == '.snapshot'):
        print('snapshot read', path)
        load_path = 'updater/model:main/'
    else:
        print('model read error')
        print(fileFuncLine())
        exit()

    return load_path


def getFilePath(folder, name, ext=''):
    """
    入力されたフォルダ名とファイル名と拡張子を連結する
    [in]  folder: 入力フォルダ名
    [in]  name:   入力ファイル名
    [in]  ext:    拡張子
    [out] 連結されたフルパスのファイル名
    """

    if not os.path.isdir(folder):
        os.makedirs(folder)

    path = os.path.join(folder, name + ext)
    print('get file path:', path)
    return path


def sortTimeStamp(folder_list, ext):
    """
    入力されたフォルダ以下のあるext拡張子のファイルをタイムスタンプでソートする
    [in]  folder_list: ファイルを探索するフォルダ
    [in]  ext:         探索するファイルの拡張子
    [out] タイムスタンプでソートされたファイルリスト
    """

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

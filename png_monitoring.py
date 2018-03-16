#!/usr/bin/env python3
# -*-coding: utf-8 -*-
#
help = '任意のフォルダの監視'
#

import os
import time
import argparse
import shutil

from func import ChangeHandler
from watchdog.observers import Observer


def command():
    parser = argparse.ArgumentParser(description=help)
    parser.add_argument('monitor', help='監視するフォルダ')
    parser.add_argument('copy', help='コピーするフォルダ')
    parser.add_argument('--force', action='store_true',
                        help='monotorとcopyのフォルダがない場合に強制的に作成する')
    return parser.parse_args()


class PNGMonitor(ChangeHandler):
    def __init__(self, copy):
        self.copy = copy

    def on_modified(self, event):
        path1, name1, ext = super().on_modified(event)
        if('png' in ext.lower()):
            time.sleep(1)
            path2 = os.path.join(self.copy, name1)
            shutil.copy2(path1, path2)


def main(monitor, copy):
    while 1:
        event_handler = PNGMonitor(copy)
        observer = Observer()
        observer.schedule(event_handler, monitor, recursive=True)
        observer.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()

        observer.join()


if __name__ in '__main__':
    args = command()
    print('Monitoring :', args.monitor)
    print('Copy to :', args.copy)
    print('Exit: Ctrl-c')

    if not os.path.isdir(args.monitor):
        if args.force:
            os.makedirs(args.monitor)
        else:
            print('[Error] monitor folder not found:', args.monitor)
            exit()

    if not os.path.isdir(args.copy):
        if args.force:
            os.makedirs(args.copy)
        else:
            print('[Error] copy folder not found:', args.copy)
            exit()

    main(args.monitor, args.copy)

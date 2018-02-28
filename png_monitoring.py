#!/usr/bin/env python3
# -*-coding: utf-8 -*-
#
help = '任意のフォルダの監視'
#

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

import os
import time
import argparse
import shutil


def command():
    parser = argparse.ArgumentParser(description=help)
    parser.add_argument('monitor', help='監視するフォルダ')
    parser.add_argument('copy', help='コピーするフォルダ')
    return parser.parse_args()


class ChangeHandler(FileSystemEventHandler):

    def __init__(self, copy):
        self.copy = copy

    # def on_created(self, event):
    #     filepath = event.src_path
    #     filename = os.path.basename(filepath)
    #     print('%sができました' % filename)

    def on_modified(self, event):
        filepath1 = event.src_path
        filename1 = os.path.basename(filepath1)
        # print('%sを変更しました' % filename)

        name, ext = os.path.splitext(filename1)
        if('png' in ext):
            time.sleep(1)
            filepath2 = os.path.join(self.copy, filename1)
            # print('{0} copy to {1}'.format(filepath1, filepath2))
            shutil.copy2(filepath1, filepath2)

    # def on_deleted(self, event):
    #     filepath = event.src_path
    #     filename = os.path.basename(filepath)
    #     print('%sを削除しました' % filename)


def main(monitor, copy):
    while 1:
        event_handler = ChangeHandler(copy)
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
        print('[Error] monitor folder not found:', args.monitor)
        exit()

    if not os.path.isdir(args.copy):
        print('[Error] copy folder not found:', args.copy)
        exit()

    main(args.monitor, args.copy)

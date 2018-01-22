#!/usr/bin/env python3
# -*-coding: utf-8 -*-
#
help = '学習メイン部'
#

import os
import json
import argparse
import numpy as np
from datetime import datetime

import chainer
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer.datasets import tuple_dataset


from Lib.network import JC
from Lib.func import argsPrint, imgs2x, img2arr
from Lib.func import getLossfun, getActfun, getFilePath, getOptimizer
from Lib.plot_report_log import PlotReportLog


def command():
    parser = argparse.ArgumentParser(description=help)
    parser.add_argument('-i', '--in_path', default='./result/',
                        help='入力データセットのフォルダ (default: ./result/)')
    parser.add_argument('-lf', '--lossfun', default='mse',
                        help='損失関数 (default: mse, other: mae)')
    parser.add_argument('-a1', '--actfun_1', default='relu',
                        help='活性化関数(1) (default: relu, other: elu, c_relu, l_relu, sigmoid, h_sigmoid, tanh, s_plus)')
    parser.add_argument('-a2', '--actfun_2', default='sigmoid',
                        help='活性化関数(2) (default: sigmoid, other: relu, elu, c_relu, l_relu, h_sigmoid, tanh, s_plus)')
    parser.add_argument('-opt', '--optimizer', default='adam',
                        help='オプティマイザ (default: adam, other: ada_d, ada_g, m_sgd, n_ag, rmsp, rmsp_g, sgd, smorms)')
    parser.add_argument('-ln', '--layer_num', type=int, default=3,
                        help='ネットワーク層の数 (default: 3)')
    parser.add_argument('-u', '--unit', type=int, default=16,
                        help='ネットワークのユニット数 (default: 16)')
    parser.add_argument('-b', '--batchsize', type=int, default=100,
                        help='ミニバッチサイズ (default: 100)')
    parser.add_argument('-e', '--epoch', type=int, default=10,
                        help='学習のエポック数 (default 10)')
    parser.add_argument('-f', '--frequency', type=int, default=-1,
                        help='スナップショット周期 (default: -1)')
    parser.add_argument('-g', '--gpu_id', type=int, default=-1,
                        help='使用するGPUのID (default -1)')
    parser.add_argument('-o', '--out_path', default='./result/',
                        help='生成物の保存先(default: ./result/)')
    parser.add_argument('-r', '--resume', default='',
                        help='使用するスナップショットのパス(default: no use)')
    parser.add_argument('--noplot', dest='plot', action='store_false',
                        help='学習過程をPNG形式で出力しない場合に使用する')
    parser.add_argument('--only_check', action='store_true',
                        help='オプション引数が正しく設定されているかチェックする')
    return parser.parse_args()


def getImgData(folder):
    for l in os.listdir(folder):
        if os.path.isdir(l):
            pass
        elif 'train.npz' in l:
            np_arr = np.load(os.path.join(folder, l))
            print('train (comp/raw): {0}/{1}'.format(
                np_arr['comp'].shape, np_arr['raw'].shape)
            )
            train = tuple_dataset.TupleDataset(
                img2arr(np_arr['comp']),
                img2arr(imgs2x(np_arr['raw']))
            )
        elif 'test.npz' in l:
            np_arr = np.load(os.path.join(folder, l))
            print('test  (comp/raw): {0}/{1}'.format(
                np_arr['comp'].shape, np_arr['raw'].shape)
            )
            test = tuple_dataset.TupleDataset(
                img2arr(np_arr['comp']),
                img2arr(imgs2x(np_arr['raw']))
            )

    return train, test


def main(args):

    # 各種データをユニークな名前で保存するために時刻情報を取得する
    now = datetime.today()
    exec_time = now.strftime('%y%m%d-%H%M%S')

    # Set up a neural network to train
    # Classifier reports softmax cross entropy loss and accuracy at every
    # iteration, which will be used by the PrintReport extension below.

    # 活性化関数を取得する
    actfun_1 = getActfun(args.actfun_1)
    actfun_2 = getActfun(args.actfun_2)
    # モデルを決定する
    model = L.Classifier(
        JC(n_unit=args.unit, layer=args.layer_num,
           actfun_1=actfun_1, actfun_2=actfun_2),
        lossfun=getLossfun(args.lossfun)
    )
    # Accuracyは今回使用しないのでFalseにする
    # もしも使用したいのであれば、自分でAccuracyを評価する関数を作成する必要あり？
    model.compute_accuracy = False

    if args.gpu_id >= 0:
        # Make a specified GPU current
        chainer.cuda.get_device_from_id(args.gpu_id).use()
        model.to_gpu()  # Copy the model to the GPU

    # Setup an optimizer
    optimizer = getOptimizer(args.optimizer)
    optimizer.setup(model)

    # Load dataset
    train, test = getImgData(args.in_path)
    # predict.pyでモデルを決定する際に必要なので記憶しておく
    model_param = {
        'unit':  args.unit,
        'img_ch': train[0][0].shape[0],
        'layer': args.layer_num,
        'actfun_1': args.actfun_1,
        'actfun_2': args.actfun_2
    }

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    # Set up a trainer
    updater = training.StandardUpdater(
        train_iter, optimizer, device=args.gpu_id)
    trainer = training.Trainer(
        updater, (args.epoch, 'epoch'), out=args.out_path)

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu_id))

    # Dump a computational graph from 'loss' variable at the first iteration
    # The "main" refers to the target link of the "main" optimizer.
    trainer.extend(
        extensions.dump_graph('main/loss', out_name=exec_time + '_graph.dot')
    )

    # Take a snapshot for each specified epoch
    frequency = args.epoch if args.frequency == -1 else max(1, args.frequency)
    trainer.extend(
        extensions.snapshot(filename=exec_time + '_{.updater.epoch}.snapshot'),
        trigger=(frequency, 'epoch')
    )

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport(log_name=exec_time + '.log'))

    # Save two plot images to the result dir
    if args.plot and extensions.PlotReport.available():
        # trainer.extend(
        #     extensions.PlotReport(['main/loss', 'validation/main/loss'],
        #                           'epoch', file_name=exec_time + '_plot.png')
        # )
        trainer.extend(
            PlotReportLog(['main/loss', 'validation/main/loss'],
                          'epoch', file_name=exec_time + '_log_plot.png')
        )

    # Print selected entries of the log to stdout
    # Here "main" refers to the target link of the "main" optimizer again, and
    # "validation" refers to the default name of the Evaluator extension.
    # Entries other than 'epoch' are reported by the Classifier link, called by
    # either the updater or the evaluator.
    trainer.extend(extensions.PrintReport([
        'epoch',
        'main/loss',
        'validation/main/loss',
        'elapsed_time'
    ]))

    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    if args.only_check is False:
        # predict.pyでモデルのパラメータを読み込むjson形式で保存する
        with open(getFilePath(args.out_path, exec_time, '.json'), 'w') as f:
            json.dump(model_param, f)

        # Run the training
        trainer.run()

        # 最後にモデルを保存する
        # スナップショットを使ってもいいが、
        # スナップショットはファイルサイズが大きいので
        chainer.serializers.save_npz(
            getFilePath(args.out_path, exec_time, '.model'),
            model
        )

    else:
        print('Check Finish:', exec_time)


if __name__ == '__main__':
    args = command()
    argsPrint(args)
    main(args)

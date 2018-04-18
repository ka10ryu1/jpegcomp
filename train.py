#!/usr/bin/env python3
# -*-coding: utf-8 -*-
#
help = '学習メイン部'
#

import json
import argparse
import numpy as np

import chainer
import chainer.links as L
from chainer import training
from chainer.training import extensions


from Lib.plot_report_log import PlotReportLog
import Tools.imgfunc as IMG
import Tools.getfunc as GET
import Tools.func as F


class ResizeImgDataset(chainer.dataset.DatasetMixin):
    def __init__(self, dataset, rate, dtype=np.float32):
        self._dataset = dataset
        self._rate = rate
        self._dtype = dtype
        self._len = len(self._dataset)

    def __len__(self):
        # データセットの数を返します
        return self._len

    def get_example(self, i):
        # データセットのインデックスを受け取って、データを返します
        inputs = self._dataset[i]
        x, y = inputs
        y = IMG.arrNx(y, self._rate)
        return x.astype(self._dtype), y.astype(self._dtype)


def command():
    parser = argparse.ArgumentParser(description=help)
    parser.add_argument('-i', '--in_path', default='./result/',
                        help='入力データセットのフォルダ [default: ./result/]')
    parser.add_argument('-n', '--network', type=int, default=0,
                        help='ネットワーク層 [default: 0(DDUU), other: 1(DUDU)]')
    parser.add_argument('-u', '--unit', type=int, default=2,
                        help='ネットワークのユニット数 [default: 2]')
    parser.add_argument('-sr', '--shuffle_rate', type=int, default=2,
                        help='PSの拡大率 [default: 2]')
    parser.add_argument('-ln', '--layer_num', type=int, default=2,
                        help='ネットワーク層の数 [default: 2]')
    parser.add_argument('-a1', '--actfun1', default='relu',
                        help='活性化関数(1) [default: relu, other: elu/c_relu/l_relu/sigmoid/h_sigmoid/tanh/s_plus]')
    parser.add_argument('-a2', '--actfun2', default='sigmoid',
                        help='活性化関数(2) [default: sigmoid, other: relu/elu/c_relu/l_relu/h_sigmoid/tanh/s_plus]')
    parser.add_argument('-d', '--dropout', type=float, default=0.0,
                        help='ドロップアウト率（0〜0.9、0で不使用）[default: 0.0]')
    parser.add_argument('-opt', '--optimizer', default='adam',
                        help='オプティマイザ [default: adam, other: ada_d/ada_g/m_sgd/n_ag/rmsp/rmsp_g/sgd/smorms]')
    parser.add_argument('-lf', '--lossfun', default='mse',
                        help='損失関数 [default: mse, other: mae, ber, gauss_kl]')
    parser.add_argument('-b', '--batchsize', type=int, default=100,
                        help='ミニバッチサイズ [default: 100]')
    parser.add_argument('-e', '--epoch', type=int, default=10,
                        help='学習のエポック数 [default 10]')
    parser.add_argument('-f', '--frequency', type=int, default=-1,
                        help='スナップショット周期 [default: -1]')
    parser.add_argument('-g', '--gpu_id', type=int, default=-1,
                        help='使用するGPUのID [default -1]')
    parser.add_argument('-o', '--out_path', default='./result/',
                        help='生成物の保存先[default: ./result/]')
    parser.add_argument('-r', '--resume', default='',
                        help='使用するスナップショットのパス[default: no use]')
    parser.add_argument('--noplot', dest='plot', action='store_false',
                        help='学習過程をPNG形式で出力しない場合に使用する')
    parser.add_argument('--only_check', action='store_true',
                        help='オプション引数が正しく設定されているかチェックする')
    return parser.parse_args()


def main(args):

    # 各種データをユニークな名前で保存するために時刻情報を取得する
    exec_time = GET.datetimeSHA()

    # Set up a neural network to train
    # Classifier reports softmax cross entropy loss and accuracy at every
    # iteration, which will be used by the PrintReport extension below.

    # 活性化関数を取得する
    actfun1 = GET.actfun(args.actfun1)
    actfun2 = GET.actfun(args.actfun2)
    # モデルを決定する
    if args.network == 0:
        from Lib.network import JC_DDUU as JC
    else:
        from Lib.network2 import JC_UDUD as JC

    model = L.Classifier(
        JC(n_unit=args.unit, layer=args.layer_num, rate=args.shuffle_rate,
           actfun1=actfun1, actfun2=actfun2, dropout=args.dropout,
           view=args.only_check),
        lossfun=GET.lossfun(args.lossfun)
    )
    # Accuracyは今回使用しないのでFalseにする
    # もしも使用したいのであれば、自分でAccuracyを評価する関数を作成する必要あり？
    model.compute_accuracy = False

    if args.gpu_id >= 0:
        # Make a specified GPU current
        chainer.cuda.get_device_from_id(args.gpu_id).use()
        model.to_gpu()  # Copy the model to the GPU
    else:
        model.to_intel64()

    # Setup an optimizer
    optimizer = GET.optimizer(args.optimizer)
    optimizer.setup(model)

    # Load dataset
    train, test = GET.imgData(args.in_path)
    train = ResizeImgDataset(train, args.shuffle_rate)
    test = ResizeImgDataset(test, args.shuffle_rate)
    # predict.pyでモデルを決定する際に必要なので記憶しておく
    model_param = {i: getattr(args, i) for i in dir(args) if not '_' in i[0]}
    model_param['shape'] = train[0][0].shape

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    # Set up a trainer
    updater = training.StandardUpdater(
        train_iter, optimizer, device=args.gpu_id
    )
    trainer = training.Trainer(
        updater, (args.epoch, 'epoch'), out=args.out_path
    )

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
    # trainer.extend(extensions.observe_lr())

    # Save two plot images to the result dir
    if args.plot and extensions.PlotReport.available():
        trainer.extend(
            PlotReportLog(['main/loss', 'validation/main/loss'],
                          'epoch', file_name='loss.png')
        )

        # trainer.extend(
        #     PlotReportLog(['lr'],
        #                   'epoch', file_name='lr.png', val_pos=(-80, -60))
        # )

    # Print selected entries of the log to stdout
    # Here "main" refers to the target link of the "main" optimizer again, and
    # "validation" refers to the default name of the Evaluator extension.
    # Entries other than 'epoch' are reported by the Classifier link, called by
    # either the updater or the evaluator.
    trainer.extend(extensions.PrintReport([
        'epoch',
        'main/loss',
        'validation/main/loss',
        # 'lr',
        'elapsed_time'
    ]))

    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    if args.only_check is False:
        # predict.pyでモデルのパラメータを読み込むjson形式で保存する
        with open(F.getFilePath(args.out_path, exec_time, '.json'), 'w') as f:
            json.dump(model_param, f, indent=4, sort_keys=True)

    # Run the training
    trainer.run()

    # 最後にモデルを保存する
    # スナップショットを使ってもいいが、
    # スナップショットはファイルサイズが大きいので
    chainer.serializers.save_npz(
        F.getFilePath(args.out_path, exec_time, '.model'),
        model
    )


if __name__ == '__main__':
    args = command()
    F.argsPrint(args)
    main(args)

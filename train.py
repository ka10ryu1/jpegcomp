#!/usr/bin/env python3
# -*-coding: utf-8 -*-
#
help = '学習メイン部'
#

import os
import argparse
import numpy as np
from datetime import datetime

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import training
from chainer.training import extensions
from chainer.datasets import tuple_dataset


from network import JC
from func import argsPrint, imgs2x, img2arr


def command():
    parser = argparse.ArgumentParser(description=help)
    parser.add_argument('-i', '--in_path', default='./result/',
                        help='入力データセットのフォルダ (default: ./result/)')
    parser.add_argument('-lf', '--lossfun', default='mse',
                        help='損失関数 (default: mse, other: mae, abs, se, softmax)')
    parser.add_argument('-a1', '--actfunc_1', default='relu',
                        help='活性化関数(1) (default: relu, other: elu, c_relu, l_relu, sigmoid, h_sigmoid, tanh, s_plus)')
    parser.add_argument('-a2', '--actfunc_2', default='sigmoid',
                        help='活性化関数(2) (default: sigmoid, other: relu, elu, c_relu, l_relu, h_sigmoid, tanh, s_plus)')
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
        if 'train' in l:
            np_arr = np.load(os.path.join(folder, l))
            train = tuple_dataset.TupleDataset(
                img2arr(np_arr['comp']),
                img2arr(imgs2x(np_arr['raw']))
            )
        elif 'test' in l:
            np_arr = np.load(os.path.join(folder, l))
            test = tuple_dataset.TupleDataset(
                img2arr(np_arr['comp']),
                img2arr(imgs2x(np_arr['raw']))
            )

    return train, test


def getLossfun(lossfun_str):
    if(lossfun_str.lower() == 'mse'):
        lossfun = F.mean_squared_error

    elif(lossfun_str.lower() == 'mae'):
        lossfun = F.mean_absolute_error

    elif(lossfun_str.lower() == 'abs'):
        lossfun = F.absolute_error

    elif(lossfun_str.lower() == 'se'):
        lossfun = F.squared_error

    elif(lossfun_str.lower() == 'softmax'):
        lossfun = F.softmax_cross_entropy

    else:
        lossfun = F.softmax_cross_entropy

    return lossfun


def getActFunc(actfunc_str):
    if(actfunc_str.lower() == 'relu'):
        actfunc = F.relu

    elif(actfunc_str.lower() == 'elu'):
        actfunc = F.elu

    elif(actfunc_str.lower() == 'c_relu'):
        actfunc = F.clipped_relu

    elif(actfunc_str.lower() == 'l_relu'):
        actfunc = F.leaky_relu

    elif(actfunc_str.lower() == 'sigmoid'):
        actfunc = F.sigmoid

    elif(actfunc_str.lower() == 'h_sigmoid'):
        actfunc = F.hard_sigmoid

    elif(actfunc_str.lower() == 'tanh'):
        actfunc = F.hard_sigmoid

    elif(actfunc_str.lower() == 's_plus'):
        actfunc = F.soft_plus

    else:
        actfunc = F.relu

    return actfunc


def main(args):

    now = datetime.today()
    exec_time = now.strftime('%y%m%d-%H%M%S')

    # Set up a neural network to train
    # Classifier reports softmax cross entropy loss and accuracy at every
    # iteration, which will be used by the PrintReport extension below.
    actfunc_1 = getActFunc(args.actfunc_1)
    actfunc_2 = getActFunc(args.actfunc_2)
    model = L.Classifier(
        JC(n_unit=args.unit, layer=args.layer_num,
           actfunc_1=actfunc_1, actfunc_2=actfunc_2),
        lossfun=getLossfun(args.lossfun)
    )
    model.compute_accuracy = False

    if args.gpu_id >= 0:
        # Make a specified GPU current
        chainer.cuda.get_device_from_id(args.gpu_id).use()
        model.to_gpu()  # Copy the model to the GPU

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # Load dataset
    train, test = getImgData(args.in_path)
    model_name = 'unit({0})_ch({1})_layer({2})_actFunc({3}_{4})_{5}.model'.format(
        args.unit, train[0][0].shape[0], args.layer_num,
        actfunc_1.__name__, actfunc_2.__name__, exec_time
    )

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    # Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu_id)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out_path)

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
        trainer.extend(
            extensions.PlotReport(['main/loss', 'validation/main/loss'],
                                  'epoch', file_name=exec_time + '_plot.png'))

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

    if not os.path.isdir(args.out_path):
        os.makedirs(args.out_path)

    if args.only_check is False:
        # Run the training
        trainer.run()
        chainer.serializers.save_npz(os.path.join(args.out_path, model_name), model)
    else:
        print('Check Finish:', exec_time)


if __name__ == '__main__':
    args = command()
    argsPrint(args)

    main(args)

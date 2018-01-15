#!/usr/bin/env python3
# -*-coding: utf-8 -*-
#
help = '学習メイン部'
#

import os
import cv2
import argparse
import numpy as np

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
    parser.add_argument('-l', '--lossfun', default='mse',
                        help='損失関数 (default: MSE)')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='ミニバッチサイズ (default: 100)')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='エポック数 (default 20)')
    parser.add_argument('--frequency', '-f', type=int, default=-1,
                        help='スナップショット周期 (default: -1)')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (default -1)')
    parser.add_argument('--out_path', '-o', default='./result/',
                        help='生成物の保存先(default: ./result/)')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=128,
                        help='Number of units(default: 128)')
    parser.add_argument('--noplot', dest='plot', action='store_false',
                        help='Disable PlotReport extension')
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

    print('lossfun:', lossfun.__name__)
    return lossfun


def main(args):

    # Set up a neural network to train
    # Classifier reports softmax cross entropy loss and accuracy at every
    # iteration, which will be used by the PrintReport extension below.
    model = L.Classifier(JC(n_size=args.unit), lossfun=getLossfun(args.lossfun))
    model.compute_accuracy = False

    if args.gpu >= 0:
        # Make a specified GPU current
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()  # Copy the model to the GPU

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # Load dataset
    train, test = getImgData(args.in_path)

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    # Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out_path)

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))

    # Dump a computational graph from 'loss' variable at the first iteration
    # The "main" refers to the target link of the "main" optimizer.
    trainer.extend(extensions.dump_graph('main/loss'))

    # Take a snapshot for each specified epoch
    frequency = args.epoch if args.frequency == -1 else max(1, args.frequency)
    trainer.extend(extensions.snapshot(), trigger=(frequency, 'epoch'))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport())

    # Save two plot images to the result dir
    if args.plot and extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(['main/loss', 'validation/main/loss'],
                                  'epoch', file_name='loss.png'))

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

    # Run the training
    trainer.run()


if __name__ == '__main__':
    args = command()
    argsPrint(args)

    main(args)

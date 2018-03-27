#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
help = 'jpegcompのネットワーク部分'
#

import time

from chainer import Chain
import chainer.initializers as I
import chainer.functions as F
import chainer.links as L


class DownSampleBlock(Chain):
    def __init__(self, n_unit, ksize, stride, pad,
                 actfun=None, dropout=0, wd=0.02):

        super(DownSampleBlock, self).__init__()
        with self.init_scope():
            self.cnv = L.Convolution2D(
                None, n_unit, ksize=ksize, stride=stride, pad=pad, initialW=I.Normal(wd)
            )
            self.brn = L.BatchRenormalization(n_unit)

        self.actfun = actfun
        self.dropout_ratio = dropout

    def __call__(self, x):
        h = self.actfun(self.brn(self.cnv(x)))
        if self.dropout_ratio > 0:
            h = F.dropout(h, self.dropout_ratio)

        return h


class UpSampleBlock(Chain):
    def __init__(self, n_unit1, n_unit2, ksize, stride, pad,
                 actfun=None, dropout=0.0, wd=0.02, rate=2):

        super(UpSampleBlock, self).__init__()
        with self.init_scope():
            self.cnv = L.Convolution2D(
                None, n_unit1, ksize=ksize, stride=stride, pad=pad, initialW=I.Normal(wd)
            )
            self.brn = L.BatchRenormalization(n_unit2)

        self.actfun = actfun
        self.dropout_ratio = dropout
        self.rate = rate

    def __call__(self, x):
        h = self.actfun(self.brn(self.PS(self.cnv(x))))
        if self.dropout_ratio > 0:
            h = F.dropout(h, self.dropout_ratio)

        return h

    def PS(self, h):
        """
        "P"ixcel"S"huffler
        Deconvolutionの高速版
        """

        batchsize, in_ch, in_h, in_w = h.shape
        out_ch = int(in_ch / (self.rate ** 2))
        out_h = in_h * self.rate
        out_w = in_w * self.rate
        out = F.reshape(h, (batchsize, self.rate, self.rate, out_ch, in_h, in_w))
        out = F.transpose(out, (0, 3, 4, 1, 5, 2))
        out = F.reshape(out, (batchsize, out_ch, out_h, out_w))
        return out


class JC_DDUU(Chain):
    def __init__(self, n_unit=128, n_out=1, rate=4,
                 layer=3, actfun1=F.relu, actfun2=F.sigmoid,
                 dropout=0.0, view=False):
        """
        [in] n_unit:    中間層のユニット数
        [in] n_out:     出力チャンネル
        [in] actfun1: 活性化関数（Layer A用）
        [in] actfun2: 活性化関数（Layer B用）
        """

        unit1 = n_unit
        unit2 = n_unit * 2
        unit4 = n_unit * 4
        unit8 = n_unit * 8
        nout = (rate**2) * n_out

        super(JC_DDUU, self).__init__()
        with self.init_scope():
            # D: n_unit, ksize, stride, pad,
            #    actfun=None, dropout=0, wd=0.02
            self.d1 = DownSampleBlock(unit1, 5, 2, 2, actfun1, dropout)
            self.d2 = DownSampleBlock(unit2, 5, 2, 2, actfun1, dropout)
            self.d3 = DownSampleBlock(unit4, 5, 2, 2, actfun1, dropout)
            self.d4 = DownSampleBlock(unit8, 5, 2, 2, actfun1, dropout)
            self.d5 = DownSampleBlock(unit8, 3, 1, 1, actfun1, dropout)

            # U: n_unit1, n_unit2, ksize, stride, pad,
            #    actfun=None, dropout=0, wd=0.02, rate=2
            self.u1 = UpSampleBlock(unit4, unit1, 5, 1, 2, actfun2, dropout)
            self.u2 = UpSampleBlock(unit4, unit1, 5, 1, 2, actfun2, dropout)
            self.u3 = UpSampleBlock(unit4, unit1, 5, 1, 2, actfun2, dropout)
            self.u4 = UpSampleBlock(unit4, unit1, 5, 1, 2, actfun2, dropout)
            self.u5 = UpSampleBlock(nout, n_out, 5, 1, 2, actfun2, 0, 0.02, rate)

        self.view = view
        self.cnt = 0
        self.timer = time.time()

        print('[Network info]', self.__class__.__name__)
        print('  Unit:\t{0}\n  Out:\t{1}\n  Drop out:\t{2}\nAct Func:\t{3}, {4}'.format(
            n_unit, n_out, dropout, actfun1.__name__, actfun2.__name__)
        )

    def block(self, f, x):
        if self.view:
            print('{0:2}: {1}\t{2:5.3f} s\t{3} '.format(
                self.cnt, f.__class__.__name__, time.time()-self.timer, x.shape))
            self.cnt += 1
            self.timer = time.time()

        return f(x)

    def __call__(self, x):
        hc = []

        ha = self.block(self.d1, x)
        hb = self.block(self.d2, ha)
        hc = self.block(self.d3, hb)
        hd = self.block(self.d4, hc)
        he = self.block(self.d5, hd)

        h = self.block(self.u1, F.concat([hd, he]))
        h = self.block(self.u2, F.concat([hc, h]))
        h = self.block(self.u3, F.concat([hb, h]))
        h = self.block(self.u4, F.concat([ha, h]))
        y = self.block(self.u5, h)

        if self.view:
            print('Output:', y.shape)
            exit()
        else:
            return y

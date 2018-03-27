#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
help = 'jpegcompのネットワーク部分'
#

from chainer import Chain
import chainer.initializers as I
import chainer.functions as F
import chainer.links as L


class DownSanpleBlock(Chain):
    def __init__(self, n_unit, ksize, stride, pad,
                 actfun=None, dropout=0, wd=0.02):

        super(DownSanpleBlock, self).__init__()
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
    def __init__(self, n_unit_1, n_unit_2, ksize, stride, pad,
                 actfun=None, dropout=0.0, wd=0.02, rate=2):

        super(UpSampleBlock, self).__init__()
        with self.init_scope():
            self.cnv = L.Convolution2D(
                None, n_unit_1, ksize=ksize, stride=stride, pad=pad, initialW=I.Normal(wd)
            )
            self.brn = L.BatchRenormalization(n_unit_2)

        self.actfun = actfun
        self.dropout_ratio = dropout
        self.rate = rate

    def __call__(self, x):
        h = self.actfun(self.brn(self.PS(self.cnv(x), self.rate)))
        if self.dropout_ratio > 0:
            h = F.dropout(h, self.dropout_ratio)

        return h

    def PS(self, h, r=2):
        """
        "P"ixcel"S"huffler
        Deconvolutionの高速版
        """

        batchsize, in_ch, in_h, in_w = h.shape
        out_ch = int(in_ch / (r ** 2))
        out_h = in_h * r
        out_w = in_w * r
        out = F.reshape(h, (batchsize, r, r, out_ch, in_h, in_w))
        out = F.transpose(out, (0, 3, 4, 1, 5, 2))
        out = F.reshape(out, (batchsize, out_ch, out_h, out_w))
        return out


class JC_DDUU(Chain):
    def __init__(self, n_unit=128, n_out=1, rate=4,
                 layer=3, actfun_1=F.relu, actfun_2=F.sigmoid,
                 dropout=0.0, view=False):
        """
        [in] n_unit:    中間層のユニット数
        [in] n_out:     出力チャンネル
        [in] layer:     中間層の数
        [in] actfun_1: 活性化関数（Layer A用）
        [in] actfun_2: 活性化関数（Layer B用）
        """

        unit1 = n_unit
        unit2 = n_unit * 2
        unit4 = n_unit * 4
        unit8 = n_unit * 8
        nout = (rate**2) * 3

        super(JC_DDUU, self).__init__()
        with self.init_scope():
            # D: n_unit, ksize, stride, pad,
            #    actfun=None, dropout=0, wd=0.02
            self.d1 = DownSanpleBlock(unit1, 5, 2, 2, actfun_1, dropout)
            self.d2 = DownSanpleBlock(unit2, 5, 2, 2, actfun_1, dropout)
            self.d3 = DownSanpleBlock(unit4, 5, 2, 2, actfun_1, dropout)
            self.d4 = DownSanpleBlock(unit8, 5, 2, 2, actfun_1, dropout)
            self.d5 = DownSanpleBlock(unit8, 3, 1, 1, actfun_1, dropout)

            # U: n_unit_1, n_unit_2, ksize, stride, pad,
            #    actfun=None, dropout=0, rate=2
            self.u1 = UpSampleBlock(unit4, unit1, 5, 1, 2, actfun_2, dropout)
            self.u2 = UpSampleBlock(unit4, unit1, 5, 1, 2, actfun_2, dropout)
            self.u3 = UpSampleBlock(unit4, unit1, 5, 1, 2, actfun_2, dropout)
            self.u4 = UpSampleBlock(unit4, unit1, 5, 1, 2, actfun_2, dropout)
            self.u5 = UpSampleBlock(nout, 3, 5, 1, 2, actfun_2, 0, rate)

        self.view = view

        print('[Network info]', self.__class__.__name__)
        print('  Unit:\t{0}\n  Out:\t{1}\n  Layer:\t{2}\n  Drop out:\t{3}\nAct Func:\t{4}, {5}'.format(
            n_unit, n_out, layer, dropout, actfun_1.__name__, actfun_2.__name__)
        )

    def block(self, f, x):
        if self.view:
            print(x.shape)

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
            print(y.shape)
            exit()

        else:
            return y

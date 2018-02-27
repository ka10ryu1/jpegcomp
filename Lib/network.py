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
                 actfun=None, wd=0.02, rate=2):

        super(UpSampleBlock, self).__init__()
        with self.init_scope():
            self.cnv = L.Convolution2D(
                None, n_unit_1, ksize=ksize, stride=stride, pad=pad, initialW=I.Normal(wd)
            )
            self.brn = L.BatchRenormalization(n_unit_2)

        self.actfun = actfun
        self.rate = rate

    def __call__(self, x):
        return self.actfun(self.brn(self.PS(self.cnv(x), self.rate)))

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
        # unit2 = max([n_unit//2, 1])
        # unit4 = max([n_unit//4, 1])
        # unit8 = max([n_unit//8, 1])
        unit2 = n_unit*2
        unit4 = n_unit*4
        unit8 = n_unit*8

        super(JC_DDUU, self).__init__()
        with self.init_scope():
            # D: n_unit, ksize, stride, pad, actfun=None, dropout=0, wd=0.02
            self.block1a = DownSanpleBlock(unit1, 5, 2, 2, actfun_1)
            self.block1b = DownSanpleBlock(unit2, 5, 2, 2, actfun_1)
            self.block1c = DownSanpleBlock(unit4, 5, 2, 2, actfun_1)
            self.block1d = DownSanpleBlock(unit8, 5, 2, 2, actfun_1)
            self.block1e = DownSanpleBlock(unit8, 3, 1, 1, actfun_1)

            # U: n_unit_1, n_unit_2, ksize, stride, pad, actfun=None, rate=2
            self.block2a = UpSampleBlock(unit4, unit1, 5, 1, 2, actfun_2)
            self.block2b = UpSampleBlock(unit4, unit1, 5, 1, 2, actfun_2)
            self.block2c = UpSampleBlock(unit4, unit1, 5, 1, 2, actfun_2)
            self.block2d = UpSampleBlock(unit4, unit1, 5, 1, 2, actfun_2)

            self.blockN = UpSampleBlock(rate**2, 1, 5, 1, 2, actfun_2, rate)

        self.layer = layer
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

        ha = self.block(self.block1a, x)
        hb = self.block(self.block1b, ha)
        hc = self.block(self.block1c, hb)
        hd = self.block(self.block1d, hc)
        he = self.block(self.block1e, hd)

        h = self.block(self.block2a, F.concat([hd, he]))
        h = self.block(self.block2b, F.concat([hc, h]))
        h = self.block(self.block2c, F.concat([hb, h]))
        h = self.block(self.block2d, F.concat([ha, h]))
        y = self.block(self.blockN, h)

        if self.view:
            print(y.shape)
            exit()

        else:
            return y

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
help = 'jpegcompのネットワーク部分その2'
#

from chainer import Chain
import chainer.functions as F
import chainer.links as L


class DownSanpleBlock(Chain):
    def __init__(self, n_unit, ksize, stride, pad, actfun=None, dropout=0):
        super(DownSanpleBlock, self).__init__()
        with self.init_scope():
            self.cnv = L.Convolution2D(
                None, n_unit, ksize=ksize, stride=stride, pad=pad
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
    def __init__(self, n_unit_1, n_unit_2, ksize, stride, pad, actfun=None, rate=2):
        super(UpSampleBlock, self).__init__()
        with self.init_scope():
            self.cnv = L.Convolution2D(
                None, n_unit_1, ksize=ksize, stride=stride, pad=pad
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


class JC(Chain):
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
        unit2 = n_unit//2
        unit4 = n_unit//4

        super(JC, self).__init__()
        with self.init_scope():
            self.block1a = DownSanpleBlock(unit2, 3, 1, 1, actfun_1)
            self.block1b = DownSanpleBlock(unit1, 5, 2, 2, actfun_1, dropout)
            self.block1c = UpSampleBlock(unit1, unit4, 5, 1, 2, actfun_2)
            if(layer > 2):
                self.block2a = DownSanpleBlock(unit2, 3, 1, 1, actfun_1)
                self.block2b = DownSanpleBlock(unit1, 5, 2, 2, actfun_1, dropout)
                self.block2c = UpSampleBlock(unit1, unit4, 5, 1, 2, actfun_2)

            if(layer > 3):
                self.block3a = DownSanpleBlock(unit2, 3, 1, 1, actfun_1)
                self.block3b = DownSanpleBlock(unit1, 5, 2, 2, actfun_1, dropout)
                self.block3c = UpSampleBlock(unit1, unit4, 5, 1, 2, actfun_2)

            if(layer > 4):
                self.block4a = DownSanpleBlock(unit2, 3, 1, 1, actfun_1)
                self.block4b = DownSanpleBlock(unit1, 5, 2, 2, actfun_1, dropout)
                self.block4c = UpSampleBlock(unit1, unit4, 5, 1, 2, actfun_2)

            if(layer > 5):
                self.block5a = DownSanpleBlock(unit2, 3, 1, 1, actfun_1)
                self.block5b = DownSanpleBlock(unit1, 5, 2, 2, actfun_1, dropout)
                self.block5c = UpSampleBlock(unit1, unit4, 5, 1, 2, actfun_2)

            self.blockNa = DownSanpleBlock(unit1, 3, 1, 1, actfun_1)
            self.blockNb = DownSanpleBlock(unit1, 3, 1, 1, actfun_1, dropout)
            self.blockNc = UpSampleBlock(rate**2, 1, 5, 1, 2, actfun_2, rate)

        self.layer = layer
        self.view = view

        print('[Network info]', self.__class__.__name__)
        print('  Unit:\t{0}\n  Out:\t{1}\n  Layer:\t{2}\n  Drop out:\t{3}\nAct Func:\t{4}, {5}'.format(
            n_unit, n_out, layer, dropout, actfun_1.__name__, actfun_2.__name__)
        )

    def block(self, a, b, c, x):
        if self.view:
            print('D', x.shape)

        h = a(x)
        if self.view:
            print('D', h.shape)

        h = b(h)
        if self.view:
            print('U', h.shape)

        h = c(h)
        return h

    def __call__(self, x):
        hc = []
        h = self.block(self.block1a, self.block1b, self.block1c, x)
        hc.append(h)

        if(self.layer > 2):
            h = self.block(self.block2a, self.block2b, self.block2c, h)
            hc.append(h)

        if(self.layer > 3):
            h = self.block(self.block3a, self.block3b, self.block3c, h)
            hc.append(h)

        if(self.layer > 4):
            h = self.block(self.block4a, self.block4b, self.block4c, h)
            hc.append(h)

        if(self.layer > 5):
            h = self.block(self.block5a, self.block5b, self.block5c, h)
            hc.append(h)

        h = F.concat(hc)
        y = self.block(self.blockNa, self.blockNb, self.blockNc, h)
        if self.view:
            print('Y', y.shape)
            exit()

        return y

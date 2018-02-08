#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
help = 'jpegcompのネットワーク部分その2'
#

from chainer import Chain
import chainer.functions as F
import chainer.links as L


class DownSanpleBlock(Chain):
    def __init__(self, n_unit, ksize, stride, pad, actfun):
        super(DownSanpleBlock, self).__init__()
        with self.init_scope():
            self.cnv = L.Convolution2D(
                None, n_unit, ksize=ksize, stride=stride, pad=pad
            )
            self.brn = L.BatchRenormalization(n_unit)

        self.actfun = actfun

    def __call__(self, x, view=False):
        if view:
            print('D', x.shape)

        return self.actfun(self.brn(self.cnv(x)))


class UpSampleBlock(Chain):
    def __init__(self, n_unit_1, n_unit_2, ksize, stride, pad, actfun, rate=2):
        super(UpSampleBlock, self).__init__()
        with self.init_scope():
            self.cnv = L.Convolution2D(
                None, n_unit_1, ksize=ksize, stride=stride, pad=pad
            )
            self.brn = L.BatchRenormalization(n_unit_2)

        self.actfun = actfun
        self.rate = rate

    def __call__(self, x, view=False):
        if view:
            print('U', x.shape)

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
                 layer=3, actfun_1=F.relu, actfun_2=F.sigmoid, view=False):
        """
        [in] n_unit:    中間層のユニット数
        [in] n_out:     出力チャンネル
        [in] layer:     中間層の数
        [in] actfun_1: 活性化関数（Layer A用）
        [in] actfun_2: 活性化関数（Layer B用）
        """

        super(JC, self).__init__()
        with self.init_scope():
            self.block1a = DownSanpleBlock(n_unit//2, 3, 1, 1, actfun_1)
            self.block1b = DownSanpleBlock(n_unit,    5, 2, 2, actfun_1)
            self.block1c = UpSampleBlock(n_unit*2, n_unit//2, 5, 1, 2, actfun_2)
            if(layer > 2):
                self.block2a = DownSanpleBlock(n_unit//2, 3, 1, 1, actfun_1)
                self.block2b = DownSanpleBlock(n_unit,    5, 2, 2, actfun_1)
                self.block2c = UpSampleBlock(n_unit*2, n_unit//2, 5, 1, 2, actfun_2)

            if(layer > 3):
                self.block3a = DownSanpleBlock(n_unit//2, 3, 1, 1, actfun_1)
                self.block3b = DownSanpleBlock(n_unit,    5, 2, 2, actfun_1)
                self.block3c = UpSampleBlock(n_unit*2, n_unit//2, 5, 1, 2, actfun_2)

            self.blockNa = DownSanpleBlock(n_unit, 3, 1, 1, actfun_1)
            self.blockNb = DownSanpleBlock(n_unit, 3, 1, 1, actfun_1)
            self.blockNc = UpSampleBlock(rate**2, 1, 5, 1, 2, actfun_2, rate)

        self.layer = layer
        self.view = view

        print('[Network info]')
        print('  Unit:\t{0}\n  Out:\t{1}\n  Layer:\t{2}\n  Act Func:\t{3}, {4}'.format(
            n_unit, n_out, layer, actfun_1.__name__, actfun_2.__name__)
        )

    def __call__(self, x):
        hc = []
        h = self.block1a(x, self.view)
        h = self.block1b(h, self.view)
        h = self.block1c(h, self.view)
        hc.append(h)

        if(self.layer > 2):
            h = self.block2a(h, self.view)
            h = self.block2b(h, self.view)
            h = self.block2c(h, self.view)
            hc.append(h)

        if(self.layer > 3):
            h = self.block3a(h, self.view)
            h = self.block3b(h, self.view)
            h = self.block3c(h, self.view)
            hc.append(h)

        h = self.blockNa(F.concat(hc), self.view)
        h = self.blockNb(h, self.view)
        y = self.blockNc(h, self.view)

        if self.view:
            print('Y', y.shape)
            exit()

        return y

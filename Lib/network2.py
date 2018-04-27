#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
help = 'jpegcompのネットワーク部分その2'
#

from chainer import Chain
import chainer.functions as F
from Lib.network import DownSampleBlock, UpSampleBlock


class JC_UDUD(Chain):
    def __init__(self, n_unit=128, n_out=1, rate=4,
                 layer=3, actfun1=F.relu, actfun2=F.sigmoid,
                 dropout=0.0, view=False):
        """
        [in] n_unit:    中間層のユニット数
        [in] n_out:     出力チャンネル
        [in] layer:     中間層の数
        [in] actfun1: 活性化関数（Layer A用）
        [in] actfun2: 活性化関数（Layer B用）
        """

        unit1 = n_unit
        unit2 = n_unit*2
        unit4 = n_unit*4

        super(JC_UDUD, self).__init__()
        with self.init_scope():
            self.d1a = DownSampleBlock(unit1, 5, 2, 2, actfun1)
            self.d1b = DownSampleBlock(unit2, 3, 1, 1, actfun1, dropout)
            self.u1c = UpSampleBlock(unit4, unit1, 5, 1, 2, actfun2)
            if(layer > 2):
                self.d2a = DownSampleBlock(unit1, 5, 2, 2, actfun1)
                self.d2b = DownSampleBlock(unit2, 3, 1, 1, actfun1, dropout)
                self.u2c = UpSampleBlock(unit4, unit1, 5, 1, 2, actfun2)

            if(layer > 3):
                self.d3a = DownSampleBlock(unit1, 5, 2, 2, actfun1)
                self.d3b = DownSampleBlock(unit2, 3, 1, 1, actfun1, dropout)
                self.u3c = UpSampleBlock(unit4, unit1, 5, 1, 2, actfun2)

            if(layer > 4):
                self.d4a = DownSampleBlock(unit1, 5, 2, 2, actfun1)
                self.d4b = DownSampleBlock(unit2, 3, 1, 1, actfun1, dropout)
                self.u4c = UpSampleBlock(unit4, unit1, 5, 1, 2, actfun2)

            if(layer > 5):
                self.da = DownSampleBlock(unit1, 5, 2, 2, actfun1)
                self.d5b = DownSampleBlock(unit2, 3, 1, 1, actfun1, dropout)
                self.d5c = UpSampleBlock(unit4, unit1, 5, 1, 2, actfun2)

            self.dNa = DownSampleBlock(unit4, 5, 2, 2, actfun1)
            self.dNb = UpSampleBlock(unit4, unit1, 3, 1, 1, actfun2)
            self.uNc = UpSampleBlock(rate**2, 1, 5, 1, 2, actfun2, rate=rate)

        self.layer = layer
        self.view = view

        print('[Network info]', self.__class__.__name__)
        print('  Unit:\t{0}\n  Out:\t{1}\n  Layer:\t{2}\n  Drop out:\t{3}\nAct Func:\t{4}, {5}'.format(
            n_unit, n_out, layer, dropout, actfun1.__name__, actfun2.__name__)
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
        h = self.block(self.d1a, self.d1b, self.u1c, x)
        hc.append(h)

        if(self.layer > 2):
            h = self.block(self.d2a, self.d2b, self.u2c, h)
            hc.append(h)

        if(self.layer > 3):
            h = self.block(self.d3a, self.d3b, self.u3c, h)
            hc.append(h)

        if(self.layer > 4):
            h = self.block(self.d4a, self.d4b, self.u4c, h)
            hc.append(h)

        if(self.layer > 5):
            h = self.block(self.d5a, self.d5b, self.u5c, h)
            hc.append(h)

        h = F.concat(hc)
        y = self.block(self.dNa, self.dNb, self.uNc, h)
        if self.view:
            print('Y', y.shape)
            exit()

        return y

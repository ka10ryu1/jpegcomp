#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
help = 'jpegcompのネットワーク部分その2'
#

from chainer import Chain
import chainer.functions as F
from Lib.network import DownSanpleBlock, UpSampleBlock


class JC_UDUD(Chain):
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
        unit2 = n_unit*2
        unit4 = n_unit*4

        super(JC_UDUD, self).__init__()
        with self.init_scope():
            self.block1a = DownSanpleBlock(unit1, 5, 2, 2, actfun_1)
            self.block1b = DownSanpleBlock(unit2, 3, 1, 1, actfun_1, dropout)
            self.block1c = UpSampleBlock(unit4, unit1, 5, 1, 2, actfun_2)
            if(layer > 2):
                self.block2a = DownSanpleBlock(unit1, 5, 2, 2, actfun_1)
                self.block2b = DownSanpleBlock(unit2, 3, 1, 1, actfun_1, dropout)
                self.block2c = UpSampleBlock(unit4, unit1, 5, 1, 2, actfun_2)

            if(layer > 3):
                self.block3a = DownSanpleBlock(unit1, 5, 2, 2, actfun_1)
                self.block3b = DownSanpleBlock(unit2, 3, 1, 1, actfun_1, dropout)
                self.block3c = UpSampleBlock(unit4, unit1, 5, 1, 2, actfun_2)

            if(layer > 4):
                self.block4a = DownSanpleBlock(unit1, 5, 2, 2, actfun_1)
                self.block4b = DownSanpleBlock(unit2, 3, 1, 1, actfun_1, dropout)
                self.block4c = UpSampleBlock(unit4, unit1, 5, 1, 2, actfun_2)

            if(layer > 5):
                self.block5a = DownSanpleBlock(unit1, 5, 2, 2, actfun_1)
                self.block5b = DownSanpleBlock(unit2, 3, 1, 1, actfun_1, dropout)
                self.block5c = UpSampleBlock(unit4, unit1, 5, 1, 2, actfun_2)

            self.blockNa = DownSanpleBlock(unit4, 5, 2, 2, actfun_1)
            self.blockNb = UpSampleBlock(unit4, unit1, 3, 1, 1, actfun_2)
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

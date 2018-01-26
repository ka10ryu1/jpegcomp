#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
help = 'jpegcompのネットワーク部分'
#

from chainer import Chain
import chainer.functions as F
import chainer.links as L


class JC(Chain):
    def __init__(self,
                 n_unit=128, n_out=1,
                 layer=3, actfun_1=F.relu, actfun_2=F.sigmoid):
        """
        [in] n_unit:    中間層のユニット数
        [in] n_out:     出力チャンネル
        [in] layer:     中間層の数
        [in] actfun_1: 活性化関数
        [in] actfun_2: 活性化関数（最終段に使用する）
        """

        super(JC, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(
                None, n_unit, ksize=3, stride=1, pad=1
            )
            self.bn1 = L.BatchNormalization(n_unit)
            self.conv2 = L.Convolution2D(
                None, n_unit, ksize=7, stride=1, pad=3
            )
            self.bn2 = L.BatchNormalization(n_unit)
            if(layer > 3):
                self.conv3 = L.Convolution2D(
                    None, n_unit, ksize=3, stride=1, pad=1
                )
                self.bn3 = L.BatchNormalization(n_unit)

            if(layer > 4):
                self.conv4 = L.Convolution2D(
                    None, n_unit, ksize=3, stride=1, pad=1
                )
                self.bn4 = L.BatchNormalization(n_unit)

            self.convN = L.Convolution2D(
                None, 4, ksize=5,  stride=1, pad=2
            )
            self.bnN = L.BatchNormalization(1)

        self.layer = layer
        self.actfun_1 = actfun_1
        self.actfun_2 = actfun_2

        print('[Network info]')
        print('  Unit:\t{0}\n  Out:\t{1}\n  Layer:\t{2}\n  Act Func:\t{3}, {4}'.format(
            n_unit, n_out, layer, actfun_1.__name__, actfun_2.__name__)
        )

    def __call__(self, x):
        h = self.actfun_1(self.bn1(self.conv1(x)))
        h = self.actfun_1(self.bn2(self.conv2(h)))
        if(self.layer > 3):
            h = self.actfun_1(self.bn3(self.conv3(h)))

        if(self.layer > 4):
            h = self.actfun_1(self.bn4(self.conv4(h)))

        y = self.actfun_2(self.bnN(self.PS(self.convN(h))))
        return y

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

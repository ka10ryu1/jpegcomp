#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
help = 'jpegcompのネットワーク部分その2'
#

from chainer import Chain
import chainer.functions as F
import chainer.links as L


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
            self.cnv1a = L.Convolution2D(None, n_unit//2, ksize=3, stride=1, pad=1)
            self.brn1a = L.BatchRenormalization(n_unit//2)
            self.cnv1b = L.Convolution2D(None, n_unit, ksize=5, stride=2, pad=2)
            self.brn1b = L.BatchRenormalization(n_unit)
            self.cnv1c = L.Convolution2D(None, n_unit*2, ksize=5,  stride=1, pad=2)
            self.brn1c = L.BatchRenormalization(n_unit//2)
            if(layer > 2):
                self.cnv2a = L.Convolution2D(None, n_unit//2, ksize=3, stride=1, pad=1)
                self.brn2a = L.BatchRenormalization(n_unit//2)
                self.cnv2b = L.Convolution2D(None, n_unit, ksize=5, stride=2, pad=2)
                self.brn2b = L.BatchRenormalization(n_unit)
                self.cnv2c = L.Convolution2D(None, n_unit*2, ksize=5,  stride=1, pad=2)
                self.brn2c = L.BatchRenormalization(n_unit//2)

            if(layer > 3):
                self.cnv3a = L.Convolution2D(None, n_unit//2, ksize=3, stride=1, pad=1)
                self.brn3a = L.BatchRenormalization(n_unit//2)
                self.cnv3b = L.Convolution2D(None, n_unit, ksize=5, stride=2, pad=2)
                self.brn3b = L.BatchRenormalization(n_unit)
                self.cnv3c = L.Convolution2D(None, n_unit*2, ksize=5,  stride=1, pad=2)
                self.brn3c = L.BatchRenormalization(n_unit//2)

            self.cnvNa = L.Convolution2D(None, n_unit, ksize=3, stride=1, pad=1)
            self.brnNa = L.BatchRenormalization(n_unit)
            self.cnvNb = L.Convolution2D(None, n_unit, ksize=3, stride=1, pad=1)
            self.brnNb = L.BatchRenormalization(n_unit)
            self.cnvNc = L.Convolution2D(None, rate**2, ksize=5,  stride=1, pad=2)
            self.brnNc = L.BatchRenormalization(1)

        self.layer = layer
        self.actfun_1 = actfun_1
        self.actfun_2 = actfun_2
        self.rate = rate
        self.view = view

        print('[Network info]')
        print('  Unit:\t{0}\n  Out:\t{1}\n  Layer:\t{2}\n  Act Func:\t{3}, {4}'.format(
            n_unit, n_out, layer, actfun_1.__name__, actfun_2.__name__)
        )

    def __call__(self, x):
        hc = []
        h = self.layer_A(x, self.brn1a, self.cnv1a)
        h = self.layer_A(h, self.brn1b, self.cnv1b)
        h = self.layer_B(h, self.brn1c, self.cnv1c)
        hc.append(h)

        if(self.layer > 2):
            h = self.layer_A(h, self.brn2a, self.cnv2a)
            h = self.layer_A(h, self.brn2b, self.cnv2b)
            h = self.layer_B(h, self.brn2c, self.cnv2c)
            hc.append(h)

        if(self.layer > 3):
            h = self.layer_A(h, self.brn3a, self.cnv3a)
            h = self.layer_A(h, self.brn3b, self.cnv3b)
            h = self.layer_B(h, self.brn3c, self.cnv3c)
            hc.append(h)

        h = F.concat(hc)
        h = self.layer_A(h, self.brnNa, self.cnvNa)
        h = self.layer_A(h, self.brnNb, self.cnvNb)
        y = self.layer_B(h, self.brnNc, self.cnvNc, r=self.rate)
        if self.view:
            print(y.shape)
            exit()

        return y

    def layer_A(self, x, brn, cnv):
        if self.view:
            print('layer A:', x.shape)

        return self.actfun_1(brn(cnv(x)))

    def layer_B(self, x, brn, cnv, r=2):
        if self.view:
            print('layer B:', x.shape)

        return self.actfun_2(brn(self.PS(cnv(x), r)))

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

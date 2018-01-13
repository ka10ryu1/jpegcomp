#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
help = 'jpegcompのネットワーク部分'
#

from chainer import Chain
import chainer.functions as F
import chainer.links as L


class JC(Chain):
    def __init__(self, n_in=1, n_size=128, n_out=1):
        """
        [in] n_in:    入力チャンネル
        [in] n_size:  中間チャンネルサイズ
        [in] n_out:   出力チャンネル
        """
        super(JC, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(n_in, n_size, 9,  pad=4)
            self.bn1 = L.BatchNormalization(n_size)
            self.conv2 = L.Convolution2D(None, n_size, 1)
            self.bn2 = L.BatchNormalization(n_size)
            self.convN = L.Convolution2D(None, 4, 5,  pad=2)
            self.bnN = L.BatchNormalization(1)

        self.i = n_in
        self.u = n_size
        print('in_ch:{0} / size:{1} / out_c:{2}'.format(n_in, n_size, n_out))

    def __call__(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        y = F.sigmoid(self.bnN(self.PS(self.convN(h))))
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

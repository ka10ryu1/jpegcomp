#!/usr/bin/env python3
# -*-coding: utf-8 -*-
#
help = 'getfuncのテスト用コード'
#

import logging
# basicConfig()は、 debug()やinfo()を最初に呼び出す"前"に呼び出すこと
logging.basicConfig(format='%(message)s')
level = logging.INFO
logging.getLogger('Tools').setLevel(level=level)

import unittest

import getfunc as GET


class TestGetFunc(unittest.TestCase):

    def test_lossfun(self):
        self.assertEqual(
            GET.lossfun('mse').__name__, 'mean_squared_error'
        )
        self.assertEqual(
            GET.lossfun('mae').__name__, 'mean_absolute_error'
        )
        self.assertEqual(
            GET.lossfun('ber').__name__, 'bernoulli_nll'
        )
        self.assertEqual(
            GET.lossfun('gauss_kl').__name__, 'gaussian_kl_divergence'
        )
        self.assertEqual(
            GET.lossfun('test').__name__, 'mean_squared_error'
        )
        self.assertEqual(
            GET.lossfun('').__name__, 'mean_squared_error'
        )

    def test_actfun(self):
        self.assertEqual(
            GET.actfun('relu').__name__, 'relu'
        )
        self.assertEqual(
            GET.actfun('elu').__name__, 'elu'
        )
        self.assertEqual(
            GET.actfun('c_relu').__name__, 'clipped_relu'
        )
        self.assertEqual(
            GET.actfun('l_relu').__name__, 'leaky_relu'
        )
        self.assertEqual(
            GET.actfun('sigmoid').__name__, 'sigmoid'
        )
        self.assertEqual(
            GET.actfun('h_sigmoid').__name__, 'hard_sigmoid'
        )
        self.assertEqual(
            GET.actfun('tanh').__name__, 'tanh'
        )
        self.assertEqual(
            GET.actfun('s_plus').__name__, 'softplus'
        )
        self.assertEqual(
            GET.actfun('none').__name__, 'F_None'
        )
        self.assertEqual(
            GET.actfun('test').__name__, 'relu'
        )
        self.assertEqual(
            GET.actfun('').__name__, 'relu'
        )

    def test_optimizer(self):
        self.assertEqual(
            GET.optimizer('adam').__class__.__name__, 'Adam'
        )
        self.assertEqual(
            GET.optimizer('ada_d').__class__.__name__, 'AdaDelta'
        )
        self.assertEqual(
            GET.optimizer('ada_g').__class__.__name__, 'AdaGrad'
        )
        self.assertEqual(
            GET.optimizer('m_sgd').__class__.__name__, 'MomentumSGD'
        )
        self.assertEqual(
            GET.optimizer('n_ag').__class__.__name__, 'NesterovAG'
        )
        self.assertEqual(
            GET.optimizer('rmsp').__class__.__name__, 'RMSprop'
        )
        self.assertEqual(
            GET.optimizer('rmsp_g').__class__.__name__, 'RMSpropGraves'
        )
        self.assertEqual(
            GET.optimizer('sgd').__class__.__name__, 'SGD'
        )
        self.assertEqual(
            GET.optimizer('smorms').__class__.__name__, 'SMORMS3'
        )
        self.assertEqual(
            GET.optimizer('test').__class__.__name__, 'Adam'
        )
        self.assertEqual(
            GET.optimizer('').__class__.__name__, 'Adam'
        )


if __name__ == '__main__':
    unittest.main()

#!/usr/bin/env python3
# -*-coding: utf-8 -*-
#
help = 'imgfuncのテスト用コード'
#

import cv2
import unittest

import Lib.imgfunc as IMG


class TestImgFunc(unittest.TestCase):

    def test_getCh(self):
        self.assertEqual(IMG.getCh(-1), cv2.IMREAD_UNCHANGED)
        self.assertEqual(IMG.getCh(0), cv2.IMREAD_UNCHANGED)
        self.assertEqual(IMG.getCh(1), cv2.IMREAD_GRAYSCALE)
        self.assertEqual(IMG.getCh(2), cv2.IMREAD_UNCHANGED)
        self.assertEqual(IMG.getCh(3), cv2.IMREAD_COLOR)
        self.assertEqual(IMG.getCh(4), cv2.IMREAD_UNCHANGED)
        self.assertEqual(IMG.getCh(2.5), cv2.IMREAD_UNCHANGED)

    def test_encodeDecode(self):
        lenna = cv2.imread('./Lib/Tests/Lenna.bmp')
        mandrill = cv2.imread('./Lib/Tests/Mandrill.bmp')
        self.assertEqual(len(IMG.encodeDecode([lenna, mandrill], 3)), 2)
        lenna = cv2.imread('./Lib/Tests/Lenna.bmp', IMG.getCh(1))
        mandrill = cv2.imread('./Lib/Tests/Mandrill.bmp', IMG.getCh(1))
        self.assertEqual(len(IMG.encodeDecode([lenna, mandrill], 1)), 2)

    def test_split(self):
        lenna = cv2.imread('./Lib/Tests/Lenna.bmp')
        mandrill = cv2.imread('./Lib/Tests/Mandrill.bmp')
        split = IMG.split([lenna, mandrill], 32)
        self.assertEqual(split[0].shape, (162, 32, 32, 3))

        split = IMG.split([lenna, mandrill], 32, 10)
        self.assertEqual(split[0].shape, (160, 32, 32, 3))
        split = IMG.split([lenna, mandrill], 32, 100)
        self.assertEqual(split[0].shape, (100, 32, 32, 3))
        with self.assertRaises(SystemExit):
            IMG.split([lenna, mandrill], 32, 1000)

        lenna = cv2.imread('./Lib/Tests/Lenna.bmp', IMG.getCh(1))
        mandrill = cv2.imread('./Lib/Tests/Mandrill.bmp', IMG.getCh(1))
        split = IMG.split([lenna, mandrill], 32)
        self.assertEqual(split[0].shape, (162, 32, 32))

    def test_rotate(self):
        lenna = cv2.imread('./Lib/Tests/Lenna.bmp')
        mandrill = cv2.imread('./Lib/Tests/Mandrill.bmp')
        self.assertEqual(len(IMG.rotate([lenna, mandrill])), 6)
        self.assertEqual(len(IMG.rotate([lenna, mandrill], num=-1)), 4)
        self.assertEqual(len(IMG.rotate([lenna, mandrill], num=0)), 4)
        self.assertEqual(len(IMG.rotate([lenna, mandrill], num=1)), 4)
        self.assertEqual(len(IMG.rotate([lenna, mandrill], num=2)), 6)
        self.assertEqual(len(IMG.rotate([lenna, mandrill], num=3)), 8)

    def test_imgs2arr(self):
        lenna = cv2.imread('./Lib/Tests/Lenna.bmp')
        mandrill = cv2.imread('./Lib/Tests/Mandrill.bmp')
        self.assertEqual(IMG.imgs2arr([lenna, mandrill]).shape, (2, 3, 256, 256))

        lenna = cv2.imread('./Lib/Tests/Lenna.bmp', IMG.getCh(1))
        mandrill = cv2.imread('./Lib/Tests/Mandrill.bmp', IMG.getCh(1))
        self.assertEqual(IMG.imgs2arr([lenna, mandrill]).shape, (2, 1, 256, 256))

    def test_getLossfun(self):
        self.assertEqual(IMG.getLossfun('mse').__name__, 'mean_squared_error')
        self.assertEqual(IMG.getLossfun('mae').__name__, 'mean_absolute_error')
        self.assertEqual(IMG.getLossfun('ber').__name__, 'bernoulli_nll')
        self.assertEqual(IMG.getLossfun('gauss_kl').__name__, 'gaussian_kl_divergence')
        self.assertEqual(IMG.getLossfun('test').__name__, 'mean_squared_error')
        self.assertEqual(IMG.getLossfun('').__name__, 'mean_squared_error')

    def test_getActfun(self):
        self.assertEqual(IMG.getActfun('relu').__name__, 'relu')
        self.assertEqual(IMG.getActfun('elu').__name__, 'elu')
        self.assertEqual(IMG.getActfun('c_relu').__name__, 'clipped_relu')
        self.assertEqual(IMG.getActfun('l_relu').__name__, 'leaky_relu')
        self.assertEqual(IMG.getActfun('sigmoid').__name__, 'sigmoid')
        self.assertEqual(IMG.getActfun('h_sigmoid').__name__, 'hard_sigmoid')
        self.assertEqual(IMG.getActfun('tanh').__name__, 'tanh')
        self.assertEqual(IMG.getActfun('s_plus').__name__, 'softplus')
        self.assertEqual(IMG.getActfun('test').__name__, 'relu')
        self.assertEqual(IMG.getActfun('').__name__, 'relu')

    def test_getOptimizer(self):
        self.assertEqual(IMG.getOptimizer('adam').__class__.__name__, 'Adam')
        self.assertEqual(IMG.getOptimizer('ada_d').__class__.__name__, 'AdaDelta')
        self.assertEqual(IMG.getOptimizer('ada_g').__class__.__name__, 'AdaGrad')
        self.assertEqual(IMG.getOptimizer('m_sgd').__class__.__name__, 'MomentumSGD')
        self.assertEqual(IMG.getOptimizer('n_ag').__class__.__name__, 'NesterovAG')
        self.assertEqual(IMG.getOptimizer('rmsp').__class__.__name__, 'RMSprop')
        self.assertEqual(IMG.getOptimizer('rmsp_g').__class__.__name__, 'RMSpropGraves')
        self.assertEqual(IMG.getOptimizer('sgd').__class__.__name__, 'SGD')
        self.assertEqual(IMG.getOptimizer('smorms').__class__.__name__, 'SMORMS3')
        self.assertEqual(IMG.getOptimizer('test').__class__.__name__, 'Adam')
        self.assertEqual(IMG.getOptimizer('').__class__.__name__, 'Adam')


if __name__ == '__main__':
    unittest.main()

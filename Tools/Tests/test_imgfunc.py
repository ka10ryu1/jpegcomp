#!/usr/bin/env python3
# -*-coding: utf-8 -*-
#
help = 'imgfuncのテスト用コード'
#

import cv2
import unittest

import Tools.imgfunc as IMG
import Tools.getfunc as GET

lenna_path = './Tools/Tests/Lenna.bmp'
mandrill_path = './Tools/Tests/Mandrill.bmp'


class TestImgFunc(unittest.TestCase):

    def test_getCh(self):
        self.assertEqual(IMG.getCh(-1), cv2.IMREAD_UNCHANGED)
        self.assertEqual(IMG.getCh(0), cv2.IMREAD_UNCHANGED)
        self.assertEqual(IMG.getCh(1), cv2.IMREAD_GRAYSCALE)
        self.assertEqual(IMG.getCh(2), cv2.IMREAD_UNCHANGED)
        self.assertEqual(IMG.getCh(3), cv2.IMREAD_COLOR)
        self.assertEqual(IMG.getCh(4), cv2.IMREAD_UNCHANGED)
        self.assertEqual(IMG.getCh(2.5), cv2.IMREAD_UNCHANGED)

    def test_resize(self):
        lenna = cv2.imread(lenna_path)
        mandrill = cv2.imread(mandrill_path)
        imgs = IMG.size2x([lenna, mandrill])
        self.assertEqual(imgs[0].shape, (512, 512, 3))
        self.assertEqual(imgs[1].shape, (512, 512, 3))
        self.assertEqual(IMG.resize(lenna, 2).shape, (512, 512, 3))
        self.assertEqual(IMG.resize(lenna, 1.5).shape, (384, 384, 3))
        self.assertEqual(IMG.resize(lenna, 0.5).shape, (128, 128, 3))

    def test_isImage(self):
        self.assertTrue(IMG.isImgPath(lenna_path))
        self.assertFalse(IMG.isImgPath('./Tools/Tests/Lenno.bmp'))

    def test_encodeDecode(self):
        lenna = cv2.imread(lenna_path)
        mandrill = cv2.imread(mandrill_path)
        self.assertEqual(len(IMG.encodeDecode([lenna, mandrill], 3)), 2)
        lenna = cv2.imread(lenna_path, IMG.getCh(1))
        mandrill = cv2.imread(mandrill_path, IMG.getCh(1))
        self.assertEqual(len(IMG.encodeDecode([lenna, mandrill], 1)), 2)

    def test_split(self):
        lenna = cv2.imread(lenna_path)
        mandrill = cv2.imread(mandrill_path)
        split = IMG.splitN([lenna, mandrill], 32)
        self.assertEqual(split[0].shape, (162, 32, 32, 3))

        split = IMG.splitN([lenna, mandrill], 32, 10)
        self.assertEqual(split[0].shape, (160, 32, 32, 3))
        split = IMG.splitN([lenna, mandrill], 32, 100)
        self.assertEqual(split[0].shape, (100, 32, 32, 3))
        with self.assertRaises(SystemExit):
            IMG.splitN([lenna, mandrill], 32, 1000)

        lenna = cv2.imread(lenna_path, IMG.getCh(1))
        mandrill = cv2.imread(mandrill_path, IMG.getCh(1))
        split = IMG.splitN([lenna, mandrill], 32)
        self.assertEqual(split[0].shape, (162, 32, 32))

    def test_rotate(self):
        lenna = cv2.imread(lenna_path)
        mandrill = cv2.imread(mandrill_path)
        self.assertEqual(len(IMG.rotate([lenna, mandrill])), 6)
        self.assertEqual(len(IMG.rotate([lenna, mandrill], num=-1)), 2)
        self.assertEqual(len(IMG.rotate([lenna, mandrill], num=0)), 2)
        self.assertEqual(len(IMG.rotate([lenna, mandrill], num=1)), 4)
        self.assertEqual(len(IMG.rotate([lenna, mandrill], num=2)), 6)
        self.assertEqual(len(IMG.rotate([lenna, mandrill], num=3)), 8)

    def test_imgs2arr(self):
        lenna = cv2.imread(lenna_path)
        mandrill = cv2.imread(mandrill_path)
        self.assertEqual(IMG.imgs2arr(
            [lenna, mandrill]).shape, (2, 3, 256, 256))

        lenna = cv2.imread(lenna_path, IMG.getCh(1))
        mandrill = cv2.imread(mandrill_path, IMG.getCh(1))
        self.assertEqual(IMG.imgs2arr(
            [lenna, mandrill]).shape, (2, 1, 256, 256))

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

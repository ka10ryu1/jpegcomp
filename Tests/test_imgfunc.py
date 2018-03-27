#!/usr/bin/env python3
# -*-coding: utf-8 -*-
#
help = 'imgfuncのテスト用コード'
#

import cv2
import unittest

import Tools.imgfunc as IMG

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
        l = cv2.imread(lenna_path)
        m = cv2.imread(mandrill_path)
        imgs = IMG.size2x([l, m])
        self.assertEqual(imgs[0].shape, (512, 512, 3))
        self.assertEqual(imgs[1].shape, (512, 512, 3))
        self.assertEqual(IMG.resize(l, 2).shape, (512, 512, 3))
        self.assertEqual(IMG.resize(l, 1.5).shape, (384, 384, 3))
        self.assertEqual(IMG.resize(l, 0.5).shape, (128, 128, 3))

    def test_isImgPath(self):
        self.assertTrue(IMG.isImgPath(lenna_path))
        self.assertFalse(IMG.isImgPath('./Tools/Tests/Lenno.bmp'))
        self.assertFalse(IMG.isImgPath(None))
        self.assertFalse(IMG.isImgPath(0))

    def test_encodeDecode(self):
        l = cv2.imread(lenna_path)
        m = cv2.imread(mandrill_path)
        self.assertEqual(IMG.encodeDecodeN([l, m], 3).shape, (2, 256, 256, 3))
        self.assertEqual(IMG.encodeDecodeN([l, m], 1).shape, (2, 256, 256))
        l = cv2.imread(lenna_path, IMG.getCh(1))
        m = cv2.imread(mandrill_path, IMG.getCh(1))
        self.assertEqual(IMG.encodeDecodeN([l, m], 3).shape, (2, 256, 256, 3))
        self.assertEqual(IMG.encodeDecodeN([l, m], 1).shape, (2, 256, 256))

    def test_cut(self):
        l = cv2.imread(lenna_path)
        m = cv2.imread(mandrill_path)
        self.assertEqual(IMG.cutN([l, m], 64).shape, (2, 64, 64, 3))
        self.assertEqual(IMG.cutN([l, m], 1).shape, (2, 256, 256, 3))
        self.assertEqual(IMG.cutN([l, m], 0).shape, (2, 256, 256, 3))
        self.assertEqual(IMG.cutN([l, m], -1).shape, (2, 256, 256, 3))

    def test_split(self):
        l = cv2.imread(lenna_path)
        m = cv2.imread(mandrill_path)
        imgs, split = IMG.splitSQN([l, m], 32)
        self.assertEqual(imgs.shape, (162, 32, 32, 3))
        self.assertEqual(split, (9, 9))

        with self.assertRaises(SystemExit):
            imgs, split = IMG.splitSQN([l, m], 0)

        imgs, split = IMG.splitSQN([l, m], 32, 10)
        self.assertEqual(imgs.shape, (160, 32, 32, 3))
        self.assertEqual(split, (9, 9))

        imgs, split = IMG.splitSQN([l, m], 32, 100)
        self.assertEqual(imgs.shape, (100, 32, 32, 3))
        self.assertEqual(split, (9, 9))

        with self.assertRaises(SystemExit):
            IMG.splitSQN([l, m], 32, 1000)

        l = cv2.imread(lenna_path, IMG.getCh(1))
        m = cv2.imread(mandrill_path, IMG.getCh(1))
        imgs, split = IMG.splitSQN([l, m], 32)
        self.assertEqual(imgs.shape, (162, 32, 32))
        self.assertEqual(split, (9, 9))

    def test_rotateRN(self):
        l = cv2.imread(lenna_path)
        m = cv2.imread(mandrill_path)
        imgs, angle = IMG.rotateRN([l, m], 3)
        self.assertEqual(imgs.shape, (6, 256, 256, 3))
        self.assertEqual(angle.shape, (6,))

    def test_flip(self):
        l = cv2.imread(lenna_path)
        m = cv2.imread(mandrill_path)
        self.assertEqual(IMG.flipN([l, m]).shape, (6, 256, 256, 3))
        self.assertEqual(IMG.flipN([l, m], -1).shape, (2, 256, 256, 3))
        self.assertEqual(IMG.flipN([l, m], 0).shape,  (2, 256, 256, 3))
        self.assertEqual(IMG.flipN([l, m], 1).shape,  (4, 256, 256, 3))
        self.assertEqual(IMG.flipN([l, m], 2).shape,  (6, 256, 256, 3))
        self.assertEqual(IMG.flipN([l, m], 3).shape,  (8, 256, 256, 3))

    def test_imgs2arr(self):
        l = cv2.imread(lenna_path)
        m = cv2.imread(mandrill_path)
        self.assertEqual(IMG.imgs2arr([l, m]).shape, (2, 3, 256, 256))

        l = cv2.imread(lenna_path, IMG.getCh(1))
        m = cv2.imread(mandrill_path, IMG.getCh(1))
        self.assertEqual(IMG.imgs2arr([l, m]).shape, (2, 1, 256, 256))


if __name__ == '__main__':
    unittest.main()

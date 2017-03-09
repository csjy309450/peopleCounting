#! /usr/bin/python2.7
# -*-encoding=utf-8-*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
import sys
import copy

import numpy as np
import tensorflow as tf
import cv2
from sklearn.externals import joblib

import BodyDetector.NMS as NMS
import BodyDetector.body as body
import BackgroudSegmenatation as bs

FLAGS = tf.app.flags.FLAGS

LOG = {
    'HEADER': '\033[95m',
    'BLUE': '\033[94m',
    'GREEN': '\033[92m',
    'ENDC': '\033[0m'
}

class BodyDetection:
    def __init__(self, param_dict):
        self.param_dict = param_dict
        self.shape = param_dict["input_shape"]
        self.person_tall_reg = joblib.load(param_dict["person_tall_model_path"])
        self.mybs = bs.BackgroudSegmentation(param_dict, self.person_tall_reg)
        # 初始化模型
        with tf.Graph().as_default() as g:
            # Get images
            self.images = tf.placeholder(tf.float32, shape=self.shape)
            # a = images.get_shape()

            # Build a Graph that computes the logits predictions from the inference model.
            self.logits = body.predict_inference(self.images)

            # Restore the moving average version of the learned variables for eval.
            variable_averages = tf.train.ExponentialMovingAverage(
                body.MOVING_AVERAGE_DECAY)
            variables_to_restore = variable_averages.variables_to_restore()
            saver = tf.train.Saver(variables_to_restore)
            self.sess = tf.Session()
            ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(self.sess, ckpt.model_checkpoint_path)
            else:
                print('No checkpoint file found')
                return

    def detect_single_imge(self, input_img):
        if input_img.shape[0] != self.shape[1] or input_img.shape[1] != self.shape[2]:
            input_img = cv2.resize(input_img, (self.shape[1], self.shape[2]))

        inputList = [input_img]

        predictions = self.sess.run([self.logits], feed_dict={self.images: inputList})

        if predictions[0][0, 0] > predictions[0][0, 1]:
            result = 'person'
            pred_value = predictions[0][0, 0]
            return result, pred_value
        else:
            result = 'inperson'
            return result, ''

    def comput_person_wh(self, pos, corection=(0, 0)):
        personRect = self.person_tall_reg.predict(pos)
        personSize = (int(personRect[0, 0]+corection[0]), int(personRect[0, 1]+corection[1]))
        return personSize

    def detect_full_image(self, intput_image, boudingRectList, b_display=False):
        self.RealPersonArray = np.empty((0, 4))
        ## 便利每一个bounding rect
        for boudingRect in boudingRectList:
            personsRectArray = np.empty((0, 5))
            ## 遍历bounding rect每一个像素
            for x_w in xrange(boudingRect[0], boudingRect[0] + boudingRect[2] + 1, 2):
                for x_h in xrange(boudingRect[1], boudingRect[1] + boudingRect[3] + 1, 2):
                    ## 跳过背景像素点,减少搜索区域
                    if x_w <= self.mybs.contoursMask.shape[1] and x_h <= self.mybs.contoursMask.shape[0]:
                        # print(self.mybs.contoursMask[x_h, x_w])
                        if np.all(self.mybs.contoursMask[x_h, x_w] == [0, 0, 0]):
                            continue
                    ## 计算某点person的(width,height)
                    pathSize = self.comput_person_wh([(x_w, x_h)], corection=self.param_dict["person_wh_correction"])

                    if b_display is True:
                        ## show detecting window
                        intput_image_copy = copy.deepcopy(intput_image)
                        cv2.rectangle(intput_image_copy, (int(x_w-pathSize[0]/2), int(x_h-pathSize[1]/2)),
                                      (int(x_w+pathSize[0]/2), int(x_h+pathSize[1]/2)), (0, 0, 255))
                        cv2.imshow('detecting', intput_image_copy)
                        cv2.waitKey(1)

                    ## 截取 sub rect and detection
                    subImg = cv2.getRectSubPix(intput_image, pathSize, (x_w, x_h))
                    result, pred_value = self.detect_single_imge(subImg)
                    # print(result, pred_value)
                    if result == 'person' and pred_value != '':
                        personsRectArray = np.row_stack((personsRectArray,
                                                              [int(x_w-pathSize[0]/2), int(x_h-pathSize[1]/2),
                                                               int(x_w+pathSize[0]/2), int(x_h+pathSize[1]/2), pred_value]))
            ## NMS放在每个bounding rect完成遍历之后,然后记录real person
            t_RealPersonIndex = NMS.py_cpu_nms(personsRectArray, self.param_dict["nms_thresh"])
            self.RealPersonArray = np.row_stack((self.RealPersonArray, personsRectArray[t_RealPersonIndex, 0:4]))

    def getRealPersonArray(self):
        return self.RealPersonArray
    

if __name__ == "__main__":
    body = BodyDetection()
    img_ = cv2.imread("/home/yangzheng/testData/BodyDataset/body/train/t_1.jpg")
    img = [cv2.resize(img_, (24, 24))]
    print(body.detect_single_imge(img))
#! /usr/bin/python2.7
# -*-encoding=utf-8-*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
import sys

import numpy as np
import tensorflow as tf
import cv2
import NMS

import body


FLAGS = tf.app.flags.FLAGS

LOG = {
    'HEADER': '\033[95m',
    'BLUE': '\033[94m',
    'GREEN': '\033[92m',
    'ENDC': '\033[0m'
}

def predict_once(saver, logits, pre_img, images):
  """Run predic once.

  Args:
    saver: Saver.
    top_k_op: Top K op.
  """
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/cifar10_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      return

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      predictions = sess.run([logits], feed_dict={images: pre_img})
      # str_output = LOG['GREEN'] + 'predict val: ' + str(predictions) + '\nresult = ' + result + LOG['ENDC']
      # print(str_output)

    #   summary = tf.Summary()
    #   summary.ParseFromString(sess.run(summary_op))
    #   summary.value.add(tag='Precision @ 1', simple_value=precision)
    #   summary_writer.add_summary(summary, global_step)
    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)

    if predictions[0][0, 0] > predictions[0][0, 1]:
        result = 'person'
        pred_value = predictions[0][0, 0]
        return result, pred_value
    else:
        result = 'inperson'
        return result, ''


def predict(pre_img):
  """Eval CIFAR-10 for a number of steps."""
  with tf.Graph().as_default() as g:
    # Get images and labels for CIFAR-10.
    images = tf.placeholder(tf.float32, shape=[1, 24, 24, 3])

    # a = images.get_shape()

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = body.predict_inference(images)

    # Calculate predictions.

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        body.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    predict_once(saver, logits, pre_img, images)

def bacth_predict(filePath):
  """Eval CIFAR-10 for a number of steps."""
  with tf.Graph().as_default() as g:
    # Get images and labels for CIFAR-10.
    images = tf.placeholder(tf.float32, shape=[1, 24, 24, 3])

    # a = images.get_shape()

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = body.predict_inference(images)

    # Calculate predictions.

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        body.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)
    for it in filePath:
        img = cv2.imread(it)
        if img.shape[0] == 0:
            return
        img = cv2.resize(img, (24, 24))
        img = np.array([img])  # 注：要做单个预测必须加此语句
        result, value = predict_once(saver, logits, img, images)
        print(result, value)

def bacth_predict_from_images(filePath):
  """Eval CIFAR-10 for a number of steps."""
  with tf.Graph().as_default() as g:
    # Get images and labels for CIFAR-10.
    images = tf.placeholder(tf.float32, shape=[1, 24, 24, 3])

    # a = images.get_shape()

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = body.predict_inference(images)

    # Calculate predictions.

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        body.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)
    for it in filePath:
        img = cv2.imread(it)
        if img.shape[0] == 0:
            return
        width = 9
        height = 20
        center_offset_w = int((width) / 2)
        center_offset_h = int((height) / 2)
        w_indx = int(img.shape[1] - (width) / 2)
        h_indx = int(img.shape[0] - (height) / 2)
        personBox = np.empty((0,5))
        for h in xrange(center_offset_h, h_indx):
            for w in xrange(center_offset_w, w_indx):
                sub_img_source = cv2.getRectSubPix(img, (width, height), (w, h))
                cv2.imshow('subimg', sub_img_source)
                cv2.waitKey(0)
                sub_img = cv2.resize(sub_img_source, (24, 24))
                sub_img = np.array([sub_img])  # 注：要做单个预测必须加此语句
                result, pre_value = predict_once(saver, logits, sub_img, images)
                print(result, pre_value)
                if result == 'person' and pre_value != '':
                    # cv2.rectangle(img, (w - center_offset_w, h - center_offset_h), (w + center_offset_w, h + center_offset_h), (0, 0, 255))
                    personBox = np.row_stack((personBox, [w - center_offset_w, h - center_offset_h, w + center_offset_w,
                                      h + center_offset_h, pre_value]))
                ## test
                # img_t = np.copy(img)
                # cv2.rectangle(img_t, (w - center_offset_w, h - center_offset_h),
                #               (w + center_offset_w, h + center_offset_h), (0, 0, 255))
                # cv2.imshow('img', img_t)
                # cv2.waitKey(0)
        persons = NMS.py_cpu_nms(personBox, 0.4)
        for it in persons:
            cv2.rectangle(img, (int(personBox[it,0]), int(personBox[it,1])), (int(personBox[it,2]), int(personBox[it,3])),
                          (0, 0, 255))
        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main(argv):  # pylint: disable=unused-argument
  # using conmande line
  if len(sys.argv)==1:
      ## test code
      filePath = [
          '/home/yangzheng/myPrograms/Libs_test/tensorflow/py_tensorflow_test/body/testdata/10.png'
           ]
      bacth_predict_from_images(filePath)
  else:
      filePath = sys.argv[1:]
      bacth_predict(filePath)




if __name__ == '__main__':
  tf.app.run()

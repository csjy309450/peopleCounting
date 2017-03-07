# -*-encoding=utf-8-*-
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A binary to train CIFAR-10 using a single GPU.

Accuracy:
cifar10_train.py achieves ~86% accuracy after 100K steps (256 epochs of
data) as judged by cifar10_eval.py.

Speed: With batch_size 128.

System        | Step Time (sec/batch)  |     Accuracy
------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import cifar10
import loadDateset

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/home/yangzheng/myPrograms/personsCounting/body/body_train',
                           """Directory where to write )
event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 20000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

image_path = '/home/yangzheng/testData/BodyDataset/'
list_file = ('train_images.txt', 'train_labels.txt')
g_height = cifar10.cifar10_input.IMAGE_SIZE
g_width = cifar10.cifar10_input.IMAGE_SIZE
g_bacthSize = FLAGS.batch_size
batchReader = loadDateset.BatchReader(image_path, list_file, (g_width, g_height), batchSize=g_bacthSize,
                                      _loadMode=loadDateset.LOAD_DYNAMIC_TO_MEMORY, _shuffle=True)


def train():
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default():
    global_step = tf.Variable(0, trainable=False)

    # Get images and labels for CIFAR-10. accoding to FLAGS.batch_size
    images = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, 24, 24, 3])
    labels = tf.placeholder(tf.int32, shape=[FLAGS.batch_size])

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = cifar10.inference(images)

    # Calculate loss.
    loss = cifar10.loss(logits, labels)

    # Caculate accuracy
    accuracy = cifar10.accuracy(logits, labels)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = cifar10.train(loss, global_step)

    # Create a saver.
    saver = tf.train.Saver(tf.all_variables())

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    # Build an initialization operation to run below.
    init = tf.global_variables_initializer()

    # Start running operations on the Graph.
    sess = tf.Session(config=tf.ConfigProto(
        log_device_placement=FLAGS.log_device_placement))
    sess.run(init)

    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)

    summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)
    print("Start Training:")
    for step in xrange(FLAGS.max_steps):
      start_time = time.time()

      batchReader.getBatchData()
      _, loss_value, predictions = sess.run([train_op, loss, accuracy], feed_dict={images: batchReader.getBatch_img(),
                                                                                   labels: batchReader.getBatch_label()})
      duration = time.time() - start_time

      # assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if step % 50 == 0:
        num_examples_per_step = FLAGS.batch_size
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = float(duration)
        # caculate accuracy
        true_count = np.sum(predictions)
        accuracy_value = float(true_count) / FLAGS.batch_size

        print("bacth %d" % step, )
        format_str = ('%s: step %d, loss = %.4f, accuracy = %f (%.1f examples/sec; %.3f ' 'sec/batch)')
        print (format_str % (datetime.now(), step, loss_value, accuracy_value,
                             examples_per_sec, sec_per_batch))

      # # 打印loss值和正确率
      # num_examples_per_step = FLAGS.batch_size
      # examples_per_sec = num_examples_per_step / duration
      # sec_per_batch = float(duration)
      # format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f ' 'sec/batch)')
      # print(format_str % (datetime.now(), step, loss_value,
      #                     examples_per_sec, sec_per_batch))

      # if step % 100 == 0:
      #   summary_str = sess.run(summary_op)
      #   summary_writer.add_summary(summary_str, step)

      # Save the model checkpoint periodically.
      if step % 5000 == 0 or (step + 1) == FLAGS.max_steps:
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)


def main(argv=None):  # pylint: disable=unused-argument
  #下载saifar10数据
  #删除前次的训练中间数据
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  #训练
  train()


if __name__ == '__main__':
  tf.app.run()

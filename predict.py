import numpy as np
import tensorflow as tf

with tf.Session() as sess:
    saver = tf.train.import_metagraph('./tmp/emnist_cnn_model/model.ckpt.meta')
    saver.restore(sess, './tmp/emnist_cnn_model/model.ckpt')

import tensorflow as tf
import numpy as np

filename_queue = tf.train.string_input_producer(
        ['RARI_test.txt'], shuffle=False, name='filename_queue')
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)


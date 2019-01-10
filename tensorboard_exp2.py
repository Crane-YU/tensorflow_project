import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference

INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_BODE = 500

def train(mnist):
    with tf.name_scope("input"):
        x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
        y = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-cinput')

    regularizer = tf.contrib.layers.l2_regularizer()
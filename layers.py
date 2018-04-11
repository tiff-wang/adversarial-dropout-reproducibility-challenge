'''
This file contains implementations of the different layers required to create the model used in the paper. A summary of the layers are as follows:
    Dense: A fully connected dense layer
    LeakyReLU: Similar to ReLU, but prevents zero activations by allowing negative activations to be scaled to a fraction
    Conv: A convolutional layer
    Drop: A dropout layer
    MaxPooling: A MaxPooling layer
    SoftMax: A SoftMax layer
'''

import tensorflow as tf
import numpy as np

SEED = 123456
rng = np.random.RandomState(SEED)

def __createWeights(shape, seed=None):
    w_init = tf.contrib.layers.variance_scaling_initializer(seed=seed)
    return tf.Variable(shape=shape, initializer=w_init)
    

def __createBiases(size):
    return tf.Variable(shape=[size], initializer=tf.constant_initializer(0.0))


def LeakyReLU(x, alpha=0.01):
    x = tf.nn.leaky_relu(x, alpha=alpha)
    return x


def MaxPooling(x, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME', data_format='NHWC'):
    x = tf.nn.max_pool(x, ksize, strides, padding, data_format)
    return x


def Dense(x, input_dim, output_dim, seed=None):
    W = __createWeights([input_dim, output_dim], seed) 
    b = __createBiases(output_dim) 
    x = tf.matmul(x, W) + b
    return x


def Conv2D(x, filter_size, n_channels, n_filters, strides=(1, 2, 2, 1), padding='SAME', data_format='NHWC'):
    shape = [filter_size, filter_size, n_channels, n_filters]
    W = __createWeights(shape)
    b = __createBiases(n_filters)
    x = tf.nn.conv2d(x, filter=W, strides=strides, padding=padding, data_format=data_format)
    x += b
    return x


def Dropout(x, probability):
    x = tf.nn.dropout(x, keep_prob=probability, seed=rng.randint(SEED))
    return x


def GaussianNoise(x, sigma=0.15):
    noise = tf.random_normal(shape=tf.shape(x), stddev=sigma)
    x += noise
    return x

def Add(x):
    return tf.add(x, tf.ones([5], tf.float32))


if __name__=='__main__':
    h = tf.placeholder(tf.float32)
    h1 = Add(h)
    out = GaussianNoise(h1, 0.15)

    with tf.Session() as sess:
        result = sess.run(out, feed_dict={h: np.array([1, 2, 3, 4, 5], np.float32)})
        print(result)

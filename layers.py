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


def __createWeights(shape, seed=None, name='weight'):
    w_init = tf.contrib.layers.variance_scaling_initializer(seed=seed)
    return tf.get_variable(name + '_w', shape=shape, initializer=w_init)
    

def __createBiases(size, name='bias'):
    return tf.get_variable(name + '_b', shape=[size], initializer=tf.constant_initializer(0.0))


def LeakyReLU(x, alpha=0.1):
    x = tf.nn.leaky_relu(x, alpha=alpha)
    return x


def MaxPooling(x, ksize=2, stride_length=2, padding='SAME', data_format='NHWC'):
    x = tf.nn.max_pool(x, (1, ksize, ksize, 1), (1, stride_length, stride_length, 1), padding, data_format)
    return x


def GlobalAveragePooling(x):
    x = tf.reduce_mean(x, [1, 2])
    return x


def Dense(x, input_dim, output_dim, seed=None, name='dense'):
    W = __createWeights([input_dim, output_dim], seed, name) 
    b = __createBiases(output_dim, name) 
    x = tf.matmul(x, W) + b
    return x


def Conv2D(x, filter_size, n_channels, n_filters, stride_length=1, padding='SAME', data_format='NHWC', name='conv'):
    shape = [filter_size, filter_size, n_channels, n_filters]
    W = __createWeights(shape, name=name)
    b = __createBiases(n_filters, name=name)
    x = tf.nn.conv2d(x, filter=W, strides=(1, stride_length, stride_length, 1), padding=padding, data_format=data_format)
    x += b
    return x


def Dropout(x, probability=0.5):
    x = tf.nn.dropout(x, keep_prob=probability, seed=rng.randint(SEED))
    return x


def GaussianNoise(x, sigma=0.15):
    noise = tf.random_normal(shape=tf.shape(x), stddev=sigma)
    x += noise
    return x


def SoftMax(x):
    x = tf.nn.softmax(x)
    return x


def CrossEntropyWithLogits(logits, labels):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
    return loss


# Formula: sum(p_i * log(p_i) - p_i * log(q_i))
def KLDivergenceWithLoigts(p, q):
    p_soft = SoftMax(p)
    distance = tf.reduce_sum(p_soft * tf.nn.log_softmax(p) - p_soft * tf.nn.log_softmax(q))
    return distance


if __name__=='__main__':
    x = tf.constant(np.repeat([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]], 16, axis=0), tf.float32)
    reshape = tf.reshape(x, [1, 16, 16, 1])
    h1 = Conv2D(reshape, 3, 1, 10, 1)

    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        result = sess.run(h1)
        print(result)
        print(result.shape)





import tensorflow as tf
import layers as L
import numpy as np


def firstHalf(x, conv_size=[128, 256, 512, 256,128]):
    x = L.GaussianNoise(x)
    x = L.Conv2D(x, filter_size=3, n_channels=1, n_filters=conv_size[0], padding='SAME', name='1a')
    x = L.LeakyReLU(x)
    x = L.Conv2D(x, filter_size=3, n_channels=conv_size[0], n_filters=conv_size[0], name='1b')
    x = L.LeakyReLU(x)
    x = L.Conv2D(x, filter_size=3, n_channels=conv_size[0], n_filters=conv_size[0], name='1c')
    x = L.MaxPooling(x, ksize=2, stride_length=2)
    x = L.Dropout(x, probability=0.5)
    
    x = L.Conv2D(x, filter_size=3, n_channels=conv_size[0], n_filters=conv_size[1], name='2a')
    x = L.LeakyReLU(x)
    x = L.Conv2D(x, filter_size=3, n_channels=conv_size[1], n_filters=conv_size[1], name='2b')
    x = L.LeakyReLU(x)
    x = L.Conv2D(x, filter_size=3, n_channels=conv_size[1], n_filters=conv_size[1], name='2c')
    x = L.LeakyReLU(x)
    x = L.MaxPooling(x, ksize=2, stride_length=2)

    x = L.Conv2D(x, filter_size=3, n_channels=conv_size[1], n_filters=conv_size[2], padding='VALID', name='3a')
    x = L.LeakyReLU(x)
    x = L.Conv2D(x, filter_size=1, n_channels=conv_size[2], n_filters=conv_size[3], name='3b')
    x = L.LeakyReLU(x)
    x = L.Conv2D(x, filter_size=1, n_channels=conv_size[3], n_filters=conv_size[4], name='3c')
    x = L.LeakyReLU(x)
    x = L.GlobalAveragePooling(x) 
    
    x = L.Dropout(x, probability=0.5)
    x = L.Dense(x, conv_size[4], 10)

    return x


def secondHalf(x):
    pass


if __name__=='__main__':
    x = tf.constant(np.repeat([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]], 16, axis=0), tf.float32)
    reshape = tf.reshape(x, [1, 16, 16, 1])
    out = firstHalf(reshape)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        result = sess.run(out)
        print(result.shape)

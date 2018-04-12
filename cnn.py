import tensorflow as tf
import layers as L
from buildingblocks import *
import numpy as np


'''
Returns a model without adversarial dropout
'''
def modelWithRandD(x):
    x = upperBlock(x)
    x = lowerBlock(x)
    return x


'''
Returns a model with adversarial dropout
'''
def modelWithAdD(x, y, fn_loss=L.KLDivergenceWithLogits):
    x = upperBlock(x)
    y_no_adD = lowerBlock(x)
    loss_no_adD = fn_loss(y_no_adD, y)

    # Derivative of loss fn wrt x
    DLoss = tf.gradients(loss_no_adD, [x])
    DLoss = tf.squeeze(tf.stop_gradient(DLoss)) # Stops backpropagation

    Jacobian_approx = DLoss * x
    mask = tf.ones_like(x)

    x, _ = advDropout(x, mask, Jacobian_approx)
    x = lowerBlock(x, name='fc')

    return x


'''
Create the AdD model
'''
def CreateAdDModel(x, y, learning_rate=0.001, optimizer=tf.train.AdamOptimizer, momentum=0.5, lmb=0.01):
    logit_rand = modelWithRandD(x)
    logit_rand_loss = L.CrossEntropyWithLogits(logit_rand, y)

    with tf.variable_scope(tf.get_variable_scope(), reuse=True) as scope:
        # With adversarial dropout
        logit_adD = modelWithAdD(x, y)
        logit_adD_loss = L.CrossEntropyWithLogits(logit_adD, y)

        # Total loss
        loss = logit_rand_loss + lmb * logit_adD_loss


    opt = optimizer(learning_rate=learning_rate, beta1=momentum)
    gradients = opt.compute_gradients(loss, tf.trainable_variables())
    train_op = opt.apply_gradients(gradients)

    return train_op, loss


if __name__=='__main__':
    x = tf.constant(np.repeat([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], [1, 2, 1, 3, 5, 7, 8, 5, 1, 2, 3, 4, 5, 6, 7, 8]], 16, axis=0), tf.float32)
    x_reshaped = tf.cast(tf.reshape(x, [2, 16, 16, 1]), tf.float32)
    y = tf.cast(tf.constant(np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0,0,0,0,1,0,0,0,0,0]])), tf.float32)
    train_op, loss = CreateAdDModel(x_reshaped, y)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for _ in range(20):
            w, l = sess.run([train_op, loss])
            print(np.array(l))



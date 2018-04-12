import tensorflow as tf
import layers as L
import numpy as np

'''
The model before AdD is applied
'''
def __upper(x, conv_size=[128, 256, 512, 256,128]):
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
    
    # x = L.Dropout(x, probability=0.5)
    # x = L.Dense(x, conv_size[4], 10)
    # x = L.SoftMax(x)

    return x


'''
The model after AdD is applied
'''
def __lower(x, n_in=128, n_out=10, name='fc'):
    x = L.Dense(x, n_in, n_out, name=name)
    return x;


'''
Returns a model without adversarial dropout
'''
def modelWithRandD(x):
    x = __upper(x)
    x = __lower(x)
    return x

'''
Returns a model with adversarial dropout
'''

def modelWithAdD(x, y, fn_loss=L.KLDivergenceWithLogits):
    x = __upper(x)
    y_no_adD = __lower(x)

    loss_no_adD = fn_loss(y_no_adD, y)

    # Derivative of loss fn wrt x
    DLoss = tf.gradients(loss_no_adD, [x])
    DLoss = tf.squeeze(tf.stop_gradient(DLoss)) # Stops backpropagation


    Jacobian_approx = DLoss * x
    mask = tf.ones_like(x)
    x, mask2 = adv_dropout(x, mask, Jacobian_approx)

    x = __lower(x, name='fc')

    return x


def adv_dropout(x, mask, Jacobian, sigma=0.05, dim=128):
    # y: output 
    # mask: current sampled dropout mask 
    # sigma: hyper-parameter for boundary 
    # Jabocian: Jacobian vector (gradient of divergence (or loss function))
    # dim: layer dimension 

    Jacobian = tf.reshape(Jacobian, [-1, dim])

    # mask = 0 --> -1 
    mask = 2 * mask - tf.ones_like(mask)

    adv_mask = mask 

    # extract the voxels for which the update conditions hold 
    # mask = 0 and J > 0 
    # or
    # mask = 1 and J < 1 
    abs_jac = tf.abs(Jacobian)
    temp = tf.cast(tf.greater(abs_jac, 0), tf.float32)
    temp = 2 * temp - 1 
    # interested in the cases when temp * mask = -1
    ext = tf.cast(tf.less(mask, temp), tf.float32)

    # keep the voxels that you want to update 
    candidates = abs_jac * ext 
    thres = tf.nn.top_k(candidates, int(dim * sigma * sigma)  + 1)[0][:,-1]

    targets = tf.cast(tf.greater(candidates, tf.expand_dims(thres, -1)), tf.float32)

    # get new mask 
    adv_mask = (mask - targets * 2 * mask + tf.ones_like(mask)) / 2.0

    output = adv_mask * x

    return output, adv_mask


def CreateAdDModel(x, y, learning_rate=0.001, optimizer=tf.train.AdamOptimizer, momentum=0.5, lmb=0.1):
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

    # return loss
    # return logit_adD, logit_adD_loss


if __name__=='__main__':
    x = tf.constant(np.repeat([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]], 16, axis=0), tf.float32)
    x_reshaped = tf.cast(tf.reshape(x, [1, 16, 16, 1]), tf.float32)
    y = tf.cast(tf.constant(np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]])), tf.float32)
    train_op, loss = CreateAdDModel(x_reshaped, y)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for _ in range(10):
            w, l = sess.run([train_op, loss])
            print(np.array(l))

import tensorflow as tf
import layers as L


'''
The model before AdD is applied
'''
def upperBlock(x, conv_size=[128, 256, 512, 256,128]):
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
def lowerBlock(x, n_in=128, n_out=10, name='fc'):
    x = L.Dense(x, n_in, n_out, name=name)
    return x;


'''
Apply adv dropout
'''
def advDropout(x, mask, Jacobian, sigma=0.05, dim=128):
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
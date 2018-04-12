import tensorflow as tf
import layers as L
from buildingblocks import *
import numpy as np
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.datasets import mnist

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
    x = lowerBlock(x)

    return x



def CreateBaseModel(x, y, learning_rate=0.001, optimizer=tf.train.AdamOptimizer, momentum=0.999):
    logit_rand = modelWithRandD(x)
    loss = L.CrossEntropyWithLogits(logit_rand, y)

    opt = optimizer(learning_rate=learning_rate, beta1=momentum)
    gradients = opt.compute_gradients(loss, tf.trainable_variables())
    train_op = opt.apply_gradients(gradients)

    return train_op, loss, logit_rand


'''
Create the AdD model for training
'''
def CreateAdDModel(x, y, learning_rate=0.001, optimizer=tf.train.AdamOptimizer, momentum=0.999, lmb=0.01):
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

    return train_op, loss, logit_rand


def createTestModel(x, y):
    with tf.variable_scope(tf.get_variable_scope(), reuse=True) as scope:
        logit_rand = modelWithRandD(x)
        logit_rand_loss = L.CrossEntropyWithLogits(logit_rand, y)

        return logit_rand, logit_rand_loss


def Accuracy(logits, labels):
    y_pred = tf.argmax(logits, 1)
    y_true = tf.argmax(labels, 1)
    equality = tf.equal(y_pred, y_true)
    accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))
    return accuracy


def doTraining(x_train, y_train, x_test, y_test, batch_size=128, epochs=10, baseline=False):
    # Training setup
    STEPS = len(x_train) // batch_size
    TEST_STEPS = len(x_test) // batch_size

    # Graph
    x_train_ph = tf.placeholder(tf.float32)
    x_test_ph = tf.placeholder(tf.float32)
    y_train_ph = tf.placeholder(tf.float32)
    y_test_ph = tf.placeholder(tf.float32)

    train_op, loss, logit_rand = CreateBaseModel(x_train_ph, y_train_ph) if baseline else CreateAdDModel(x_train_ph, y_train_ph)
    logit_test, test_loss = createTestModel(x_test_ph, y_test_ph)

    # Accuracy Train
    accuracy_train = Accuracy(logit_rand, y_train_ph)

    # Accuracy Test
    accuracy_test = Accuracy(logit_test, y_test_ph)

    # Training
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for epoch in range(epochs):
            # Train model
            acc_train, loss_train = 0, 0
            for i in range(STEPS):
                _, l, acc = sess.run([train_op, loss, accuracy_train], feed_dict={x_train_ph: x_train[batch_size * i : batch_size * (i + 1)], y_train_ph: y_train[batch_size * i : batch_size * (i + 1)]})
                acc_train += acc
                loss_train += l
            
            # Test model
            acc_test, loss_test = 0, 0
            for i in range(TEST_STEPS):
                acc_t, loss_t = sess.run([accuracy_test, test_loss], feed_dict={x_test_ph: x_test[batch_size * i : batch_size * (i + 1)], y_test_ph: y_test[batch_size * i : batch_size * (i + 1)]})
                acc_test += acc_t
                loss_test += loss_t

            print('Epoch: {} || Train Loss: {}, Train Acc: {} || Test Loss: {}, Test Accuracy: {}'.format(epoch, loss_train / STEPS, acc_train / STEPS, loss_test / TEST_STEPS, acc_test / TEST_STEPS))



if __name__=='__main__':
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    with tf.Session() as sess:
        x_train = sess.run(tf.image.rgb_to_grayscale(x_train))
        x_test = sess.run(tf.image.rgb_to_grayscale(x_test))
    
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    BATCH_SIZE = 128
    EPOCHS = 15
    doTraining(x_train, y_train, x_test, y_test, BATCH_SIZE, EPOCHS, False)



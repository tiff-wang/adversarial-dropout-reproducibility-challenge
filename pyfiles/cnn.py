import tensorflow as tf
import layers as L
from buildingblocks import *
import numpy as np
import re
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.datasets import mnist
import matplotlib
matplotlib.use("TkAGG")
from matplotlib import pyplot as plt

''' Preprocesses the data '''
def preprocess(data):
    # Mean normalization
    data = (data - 127.5) / 255.
    # Find principal component
    shape = data.shape
    data = data.transpose(0, 2, 3, 1)
    flatx = np.reshape(data, (data.shape[0], data.shape[1] * data.shape[2] * data.shape[3]))
    sigma = np.dot(flatx.T, flatx) / flatx.shape[1]
    U, S, V = np.linalg.svd(sigma)
    pc = np.dot(np.dot(U, np.diag(1. / np.sqrt(S + 0.0001))), U.T)
    # Apply ZCA whitening
    whitex = np.dot(flatx, pc)
    data = np.reshape(whitex, (shape[0], shape[1], shape[2], shape[3]))
    return data

'''
Uncomment to download graphs from google colab
'''
# from google.colab import files

'''
Returns a model without adversarial dropout
'''
def modelWithRandD(x, n_channels=3):
    x = upperBlock(x, n_channels=n_channels)
    x = lowerBlock(x)
    return x


'''
Returns a model with adversarial dropout
'''
def modelWithAdD(x, y, fn_loss=L.KLDivergenceWithLogits, n_channels=3):
    x = upperBlock(x, n_channels=n_channels)
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



def CreateBaseModel(x, y, learning_rate=0.001, optimizer=tf.train.AdamOptimizer, momentum=0.999, n_channels=3):
    logit_rand = modelWithRandD(x, n_channels=n_channels)
    loss = L.CrossEntropyWithLogits(logit_rand, y)

    opt = optimizer(learning_rate=learning_rate, beta1=momentum)
    gradients = opt.compute_gradients(loss, tf.trainable_variables())
    train_op = opt.apply_gradients(gradients)

    return train_op, loss, logit_rand


'''
Create the AdD model for training
'''
def CreateAdDModel(x, y, learning_rate=0.001, optimizer=tf.train.AdamOptimizer, momentum=0.999, lmb=0.01, n_channels=3):
    logit_rand = modelWithRandD(x, n_channels=n_channels)
    logit_rand_loss = L.CrossEntropyWithLogits(logit_rand, y)

    with tf.variable_scope(tf.get_variable_scope(), reuse=True) as scope:
        # With adversarial dropout
        logit_adD = modelWithAdD(x, y, n_channels=n_channels)
        logit_adD_loss = L.CrossEntropyWithLogits(logit_adD, y)

        # Total loss
        loss = logit_rand_loss + lmb * logit_adD_loss


    opt = optimizer(learning_rate=learning_rate, beta1=momentum)
    gradients = opt.compute_gradients(loss, tf.trainable_variables())
    train_op = opt.apply_gradients(gradients)        

    return train_op, loss, logit_rand


def CreateTestModel(x, y):  
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


def visualize(acc, loss, param):
    x = np.arange(0, len(acc) * param['STEPS'], param['STEPS'])
    name = "Baseline" if param['BASELINE'] else "Adversarial"
    plt.figure(1)
    plt.title("Acc trend: " + name)
    plt.plot(x, acc)
    filename = str("Accuracy_" + re.sub("{|}|:|'|,| ", "", str(param)) + ".png")
    print(filename)
    plt.savefig(filename)

    # Uncomment line to download file on Google Colab
    # files.download(filename)

    plt.figure(2)
    plt.title("Loss trend: " + name)
    plt.plot(x, loss)
    filename = str("Loss_" + re.sub("{|}|:|'|,| ", "", str(param)) + ".png")
    print(filename)
    plt.savefig(filename)

    # Uncomment line to download file on Google Colab
    # files.download(filename)

    plt.close('all')


def doTraining(x_train, y_train, x_test, y_test, param):
    # Training setup

    batch_size = param['BATCH_SIZE']
    epochs = param['EPOCHS']
    n_channels = x_train.shape[-1]

    STEPS = len(x_train) // batch_size if param['STEPS'] is None else param['STEPS']
    TEST_STEPS = len(x_test) // batch_size if param['STEPS'] is None else param['STEPS']


    # Graph
    x_train_ph = tf.placeholder(tf.float32)
    x_test_ph = tf.placeholder(tf.float32)
    y_train_ph = tf.placeholder(tf.float32)
    y_test_ph = tf.placeholder(tf.float32)

    train_op, loss, logit_rand = CreateBaseModel(x_train_ph, y_train_ph, n_channels=n_channels) if param['BASELINE'] else CreateAdDModel(
        x_train_ph, y_train_ph, n_channels=n_channels)
    logit_test, test_loss = CreateTestModel(x_test_ph, y_test_ph)

    # Accuracy Train
    accuracy_train = Accuracy(logit_rand, y_train_ph)

    # Accuracy Test
    accuracy_test = Accuracy(logit_test, y_test_ph)

    acc_train_trend = []
    loss_train_trend = []
    acc_test_trend = []
    loss_test_trend = []

    # Training
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(epochs):
            # Train model
            acc_train, loss_train = 0, 0
            for i in range(STEPS):
                _, loss_, acc = sess.run([train_op, loss, accuracy_train],
                                         feed_dict={x_train_ph: x_train[batch_size * i: batch_size * (i + 1)],
                                                    y_train_ph: y_train[batch_size * i: batch_size * (i + 1)]})
                acc_train += acc
                loss_train += loss_

                acc_train_trend.append(acc)
                loss_train_trend.append(loss_)

            # Test model
            acc_test, loss_test = 0, 0
            for i in range(TEST_STEPS):
                acc_t, loss_t = sess.run([accuracy_test, test_loss],
                                         feed_dict={x_test_ph: x_test[batch_size * i: batch_size * (i + 1)],
                                                    y_test_ph: y_test[batch_size * i: batch_size * (i + 1)]})
                acc_test += acc_t
                loss_test += loss_t

                acc_test_trend.append(acc_t)
                loss_test_trend.append(loss_t)

            print('Epoch: {} || Train Loss: {}, Train Acc: {} || Test Loss: {}, Test Accuracy: {}'.format(epoch,
                                                                                                          loss_train / STEPS,
                                                                                                          acc_train / STEPS,
                                                                                                          loss_test / TEST_STEPS,
                                                                                                          acc_test / TEST_STEPS))

    # Train
    param['TYPE'] = 'train'
    param['STEPS'] = STEPS
    visualize(acc_train_trend, loss_train_trend, param)

    # Test
    param['TYPE'] = 'test'
    param['STEPS'] = TEST_STEPS
    visualize(acc_test_trend, loss_test_trend, param)

    return acc_train_trend, loss_train_trend, acc_test_trend, loss_test_trend


if __name__=='__main__':
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # with tf.Session() as sess:
    #     x_train = sess.run(tf.image.rgb_to_grayscale(x_train))
    #     x_test = sess.run(tf.image.rgb_to_grayscale(x_test))

    x_train, x_test = preprocess(x_train), preprocess(x_test)

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    param = {
        'BATCH_SIZE': 128,
        'EPOCHS': 10,
        'STEPS': None,
        'BASELINE': False
    }
    doTraining(x_train, y_train, x_test, y_test, param)



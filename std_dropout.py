from __future__ import print_function
import numpy as np
import matplotlib 
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
from keras import backend as K
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, GaussianNoise
from keras.utils import to_categorical
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers.advanced_activations import LeakyReLU 


def supervised_model(input_shape=(32, 32, 3), stddev=0.15, dropout=0.5, alpha=0.1, adversarial=False, NUM_CLASSES=10):
	model = Sequential()
	model.add(GaussianNoise(stddev=stddev, input_shape=input_shape))

	model.add(Conv2D(128, kernel_size=(3, 3), padding="same"))
	model.add(LeakyReLU(alpha=alpha))
	model.add(Conv2D(128, kernel_size=(3, 3), padding="same"))
	model.add(LeakyReLU(alpha=alpha))
	model.add(Conv2D(128, kernel_size=(3, 3), padding="same"))
	model.add(LeakyReLU(alpha=alpha))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(dropout))

	model.add(Conv2D(256, kernel_size=(3, 3), padding="same"))
	model.add(LeakyReLU(alpha=alpha))
	model.add(Conv2D(256, kernel_size=(3, 3), padding="same"))
	model.add(LeakyReLU(alpha=alpha))
	model.add(Conv2D(256, kernel_size=(3, 3), padding="same"))
	model.add(LeakyReLU(alpha=alpha))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(512, kernel_size=(3, 3), padding="valid"))
	model.add(LeakyReLU(alpha=alpha))
	model.add(Conv2D(256, kernel_size=(1, 1)))
	model.add(LeakyReLU(alpha=alpha))
	model.add(Conv2D(128, kernel_size=(1, 1)))
	model.add(LeakyReLU(alpha=alpha))
	model.add(GlobalAveragePooling2D())

	if adversarial:
		#TODO update adversarial droput layer 
		pass
	else: 
		model.add(Dropout(dropout))

	model.add(Dense(NUM_CLASSES, activation='softmax'))

	model.compile(loss=keras.losses.categorical_crossentropy,
	              optimizer=keras.optimizers.Adam(),
	              metrics=['accuracy'])
	return model

if __name__:
	IMAGE_SIZE = 32
	NUM_CLASSES = 10
	NUM_EXAMPLES = 60000
	SPLIT = 0.10
	NUM_EXAMPLES_TRAIN = int(NUM_EXAMPLES * (1 - SPLIT))
	NUM_EXAMPLES_TEST = int(NUM_EXAMPLES * SPLIT)

	(x_train, y_train), (x_test, y_test) = cifar10.load_data()
	y_train = to_categorical(y_train)
	y_test = to_categorical(y_test)

	# x_train.shape = (50000, 32, 32, 3)
	# y_train.shape = (50000, 1)
	# x_test.shape = (10000, 32, 32, 3)
	# y_test.shape = (10000, 1)
	model = supervised_model()
	model.fit(x_train, y_train, batch_size=64, epochs=3, verbose=1, validation_data=(x_test, y_test), shuffle=True)

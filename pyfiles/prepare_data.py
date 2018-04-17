import os
import urllib.request
import tarfile
import sys
import time
import glob
import pickle
import numpy as np
from scipy import linalg
import tensorflow as tf

REMOTE_URL = r'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
LOCAL_DIR = r'/home/nabillionaire/Documents/Repositories/adversarial-dropout-project/data'

NUM_CLASSES = 10
TRAIN_EXAMPLES = 50000
TEST_EXAMPLES = 10000

''' Prints the progress during download '''
def progress(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write('\r...%d%%, %d MB, %d KB/s, %d seconds passed' %
                    (percent, progress_size / (1024 * 1024), speed, duration))
    sys.stdout.flush()

''' Downloads the data from the remote url '''
def download_data():
    print('Downloading data ....')
    if not os.path.exists(LOCAL_DIR):
        os.makedirs(LOCAL_DIR)
    file_path = os.path.join(LOCAL_DIR, 'cifar10.tar.gz')
    if not os.path.exists(file_path):
        urllib.request.urlretrieve(REMOTE_URL, file_path, progress)
        tarfile.open(file_path, 'r:gz').extractall(LOCAL_DIR)

''' Loads the data for preparation '''
def load_data():
    download_data()

    print('Loading training data .... ')
    train_images = np.zeros((TRAIN_EXAMPLES, 3 * 32 * 32), dtype=np.float32)
    train_labels = []
    for i, file in enumerate(sorted(glob.glob(LOCAL_DIR + '/cifar-10-batches-py/data_batch*'))):
        batch = pickle.load(open(file, 'rb'), encoding='latin-1')
        train_images[i * 10000:(i + 1) * 10000] = batch['data']
        train_labels.extend(batch['labels'])
    train_images = (train_images - 127.5) / 255. # use mean normalization
    train_labels = np.asarray(train_labels, dtype=np.int64)
    shuffled = np.random.permutation(TRAIN_EXAMPLES)
    train_images, train_labels = train_images[shuffled], train_labels[shuffled]

    print('Loading testing data .... ')
    test_data = pickle.load(open(LOCAL_DIR + '/cifar-10-batches-py/test_batch', 'rb'), encoding='latin-1')
    test_images = test_data['data'].astype(np.float32)
    test_images = (test_images - 127.5) / 255. # use mean normalization
    test_labels = np.asarray(test_data['labels'], dtype=np.int64)

    print('Applying ZCA whitening ....')
    mean = np.mean(train_images, axis=0)
    deviation = train_images - mean
    U, S, V = linalg.svd(np.dot(deviation.T, deviation) / deviation.shape[0])
    components = np.dot(np.dot(U, np.diag(1 / np.sqrt(S) + 1e-6)), U.T)
    np.save('{}/components'.format(LOCAL_DIR), components)
    np.save('{}/mean'.format(LOCAL_DIR), mean)
    train_images = np.dot(train_images - mean, components.T)
    test_images = np.dot(test_images - mean, components.T)

    print('Reshaping images ...')
    train_images = train_images.reshape(
        (TRAIN_EXAMPLES, 3, 32, 32)).transpose((0, 2, 3, 1)).reshape((TRAIN_EXAMPLES, -1))
    test_images = test_images.reshape(
        (TEST_EXAMPLES, 3, 32, 32)).transpose((0, 2, 3, 1)).reshape((TEST_EXAMPLES, -1))

    return (train_images, train_labels), (test_images, test_labels)

''' Writes the processed data to a file '''
def write_processed_data(images, labels, file):
    print('Writing ', file)
    writer = tf.python_io.TFRecordWriter(file)
    for index in range(labels.shape[0]):
        image = images[index].tolist()
        feature = tf.train.Feature(float_list=tf.train.FloatList(value=image))
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[32])),
            'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[32])),
            'depth': tf.train.Feature(int64_list=tf.train.Int64List(value=[3])),
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(labels[index])])),
            'image': feature
        }))
        writer.write(example.SerializeToString())
    writer.close()

''' Prepares the data for training '''
def prepare_data():
    (train_images, train_labels), (test_images, test_labels) = load_data()
    directory = os.path.join(LOCAL_DIR, 'seed' + str(1))
    if not os.path.exists(directory):
        os.makedirs(directory)
    rng = np.random.RandomState(1)
    permutation = rng.permutation(TRAIN_EXAMPLES)
    train_images, train_labels = train_images[permutation], train_labels[permutation]
    write_processed_data(train_images, train_labels, os.path.join(LOCAL_DIR, 'cifar10_train.tfrecords'))
    write_processed_data(test_images, test_labels, os.path.join(LOCAL_DIR, 'cifar10_test.tfrecords'))

def main():
    prepare_data()

if __name__ == "__main__":
    main()
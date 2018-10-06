#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
import time
import socks
import socket
import urllib.request
import urllib.parse
import numpy as np
from random import randint

import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from tensorflow.python import debug as tfdebug

""" Notice to put ```import matplotlib.pyplot``` after imports of tensorlayer, 
otherwise you will get below warning:

This call to matplotlib.use() has no effect because the backend has already
been chosen; matplotlib.use() must be called *before* pylab, matplotlib.pyplot,
or matplotlib.backends is imported for the first time.
"""
import matplotlib.pyplot as plt

# Configure your proxy setting if you are inside GW.
"""
# Create the object, assign it to a variable
_proxy_handler = urllib.request.ProxyHandler(
    {'https': 'http://127.0.0.1:8090/proxy.pac',
     'http': 'http://127.0.0.1:8090/proxy.pac'}
)
# Construct a new opener using your proxy settings
_opener = urllib.request.build_opener(_proxy_handler)
# Install the opener on the module-level
urllib.request.install_opener(_opener)
"""

# Configure your socks5 proxy setting if you are inside GW.
socks.set_default_proxy(socks.SOCKS5, "localhost", 1080)
socket.socket = socks.socksocket

tf.logging.set_verbosity(tf.logging.DEBUG)
tl.logging.set_verbosity(tl.logging.DEBUG)

"""
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    raise SystemError('*** GPU device not found ***')
print('### Found GPU at: {} ###'.format(device_name))
"""

config = tf.ConfigProto()
# See https://www.tensorflow.org/tutorials/using_gpu#allowing_gpu_memory_growth
config.gpu_options.allow_growth = True
config.allow_soft_placement = True

working_directory = 'data'
dataset_directory = 'data/quickdraw'
# categories_filename = 'categories.txt'
categories_file_url_source = 'https://raw.githubusercontent.com/googlecreativelab/quickdraw-dataset/master/'
# curl -v --socks5 127.0.0.1:1080 "https://storage.cloud.google.com/quickdraw_dataset/full/numpy_bitmap/aircraft%20carrier.npy"
# npy_dataset_url_source = 'https://storage.cloud.google.com/quickdraw_dataset/full/numpy_bitmap/'
npy_dataset_url_source = 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/'

X = "X"
y = "y"

X_NUMPY_DTYPE = np.float32
y_NUMPY_DTYPE = np.int64
X_TF_DTYPE = tf.float32
y_TF_DTYPE = tf.int32

image_height = 28
image_width = 28
image_depth = 1
image_size = image_height * image_width * image_depth
input_layer_X_shape = [image_height, image_width, image_depth]
input_layer_X_shape_batch = [-1, image_height, image_width, image_depth]

mini_categories_filename = 'web/mini-categories.txt'
n_category = 10  # The maximum of category number is up to 345
n_train_example_per_category = 20000


def print_dataset_shape(X_name, X, y_name, y):
    print(X_name + '.shape ', X.shape, end='\t,\t')
    print(y_name + '.shape ', y.shape)
    print('%s.dtype %s\t,\t%s.dtype %s' % (X_name, X.dtype, y_name, y.dtype))


def show_image(X, y, categories):
    plt.imshow(X.reshape(image_height, image_width), cmap="gray", interpolation='nearest')
    plt.title(f"{categories[y]}(label: {y})")
    plt.show()


def load_quickdraw_dataset(
    n_category=10, n_train_example_per_category=20000, plotable=False
):
    """ Download the quick draw data set. """
    n_validation_example_per_category = int(n_train_example_per_category / 0.7 * 0.2)
    n_test_example_per_category = int(n_train_example_per_category / 0.7 * 0.1)

    # Download the categories file
    # tl.files.utils.maybe_download_and_extract(categories_filename, dataset_directory, categories_file_url_source)

    tl.logging.info("Load or Download quick draw > {}".format(dataset_directory))

    train_set = {X: np.empty([0, image_size], dtype=X_NUMPY_DTYPE), y: np.empty([0], dtype=y_NUMPY_DTYPE)}
    validation_set = {X: np.empty([0, image_size], dtype=X_NUMPY_DTYPE), y: np.empty([0], dtype=y_NUMPY_DTYPE)}
    test_set = {X: np.empty([0, image_size], dtype=X_NUMPY_DTYPE), y: np.empty([0], dtype=y_NUMPY_DTYPE)}

    # category_names = [line.rstrip('\n') for line in open(f"{dataset_directory}/{categories_filename}")]
    category_names = [line.rstrip('\n') for line in open(mini_categories_filename)]
    for category_index, category_name in enumerate(category_names):
        if category_index == n_category:
            break

        category_names[category_index], _, _ = category_name.rpartition('=')
        category_name = category_names[category_index]

        filename = urllib.parse.quote(category_name) + '.npy'
        tl.files.utils.maybe_download_and_extract(filename, dataset_directory, npy_dataset_url_source)

        data = np.load(os.path.join(dataset_directory, filename))
        size_per_category = data.shape[0]
        labels = np.full(size_per_category, category_index)

        print(f"### Category '{category_name}' id:{category_index} dataset info ###")
        print_dataset_shape("data", data, "labels", labels)

        number_begin = 0
        number_end = n_train_example_per_category
        # train_set[X] = np.concatenate((train_set[X], data[number_begin: number_end, :]), axis=0)
        train_set[X] = np.vstack((train_set[X], data[number_begin: number_end, :]))
        train_set[y] = np.append(train_set[y], labels[number_begin: number_end])

        number_begin += n_train_example_per_category
        number_end += n_validation_example_per_category
        # validation_set[X] = np.concatenate((validation_set[X], data[number_begin:number_end, :]), axis=0)
        validation_set[X] = np.vstack((validation_set[X], data[number_begin:number_end, :]))
        validation_set[y] = np.append(validation_set[y], labels[number_begin:number_end])

        number_begin += n_validation_example_per_category
        number_end += n_test_example_per_category
        # test_set[X] = np.concatenate((test_set[X], data[number_begin:number_end, :]), axis=0)
        test_set[X] = np.vstack((test_set[X], data[number_begin:number_end, :]))
        test_set[y] = np.append(test_set[y], labels[number_begin:number_end])

        print_dataset_shape("train_set[X]", train_set[X], "train_set[y]", train_set[y])
        print_dataset_shape("validation_set[X]", validation_set[X], "validation_set[y]", validation_set[y])
        print_dataset_shape("test_set[X]", test_set[X], "test_set[y]", test_set[y])

    # Randomize the dataset
    size_per_set = train_set[X].shape[0]
    permutation = np.random.permutation(size_per_set)
    train_set[X] = train_set[X][permutation, :]
    train_set[y] = train_set[y][permutation]

    size_per_set = validation_set[X].shape[0]
    permutation = np.random.permutation(size_per_set)
    validation_set[X] = validation_set[X][permutation, :]
    validation_set[y] = validation_set[y][permutation]

    size_per_set = test_set[X].shape[0]
    permutation = np.random.permutation(size_per_set)
    test_set[X] = test_set[X][permutation, :]
    test_set[y] = test_set[y][permutation]

    # Reshape for CNN input
    train_set[X] = train_set[X].reshape(input_layer_X_shape_batch)
    validation_set[X] = validation_set[X].reshape(input_layer_X_shape_batch)
    test_set[X] = test_set[X].reshape(input_layer_X_shape_batch)

    # The original grayscale image is 'black background (x==0) and gray~white (0< x <=255) brush'
    # Because the CNN model doesn't need to learn the grayscale values and it only needs to
    # learn the strokes, we normalize it to 'white background (x==1) and block (x==0) brush'.
    train_set[X] = 1.0 - np.ceil(train_set[X] / 255.0)
    validation_set[X] = 1.0 - np.ceil(validation_set[X] / 255.0)
    test_set[X] = 1.0 - np.ceil(test_set[X] / 255.0)

    """
    if plotable:
        r = randint(0, n_category * n_train_example_per_category - 1)
        show_image(train_set[X][r], train_set[y][r], categories)
    
        r = randint(0, n_category * n_validation_example_per_category - 1)
        show_image(validation_set[X][r], validation_set[y][r], categories)
    
        r = randint(0, n_category * n_test_example_per_category - 1)
        show_image(test_set[X][r], test_set[y][r], categories)
    """

    return category_names, train_set, validation_set, test_set


# Open TensorBoard logs writer
tfboard_file_writer = tf.summary.FileWriter('logs')

# Download data
category_names, train_set, validation_set, test_set = load_quickdraw_dataset(n_category, n_train_example_per_category)


def model_batch_normalization(X_batch, y_batch, output_units, reuse, is_train):
    """ Define the network model """
    W_init1 = tf.truncated_normal_initializer(stddev=5e-2)
    W_init2 = tf.truncated_normal_initializer(stddev=0.04)
    bias_init = tf.constant_initializer(value=0.1)

    with tf.variable_scope("model", reuse=reuse):
        net = InputLayer(X_batch, name='input')
        net = Conv2d(net, 64, (3, 3), (1, 1), padding='SAME',
                     W_init=W_init1, b_init=None, name='cnn1')
        net = BatchNormLayer(net, is_train, act=tf.nn.relu, name='batch1')
        net = MaxPool2d(net, (3, 3), (2, 2), padding='SAME', name='pool1')

        net = Conv2d(net, 64, (3, 3), (1, 1), padding='SAME',
                     W_init=W_init1, b_init=None, name='cnn2')
        net = BatchNormLayer(net, is_train, act=tf.nn.relu, name='batch2')
        net = MaxPool2d(net, (3, 3), (2, 2), padding='SAME', name='pool2')

        net = FlattenLayer(net, name='flatten')
        net = DenseLayer(net, 384, act=tf.nn.relu,
                         W_init=W_init2, b_init=bias_init, name='d1relu')
        net = DenseLayer(net, 192, act=tf.nn.relu,
                         W_init=W_init2, b_init=bias_init, name='d2relu')
        # The softmax() is implemented internally in tl.cost.cross_entropy(y, y_) to
        # speed up computation, so we use identity here.
        # see tf.nn.sparse_softmax_cross_entropy_with_logits()
        net = DenseLayer(net, n_units=output_units, act=None,
                         W_init=W_init2, name='output')

        y_prediction_batch_without_softmax = net.outputs

        # For inference by using this model
        # y_output = tf.argmax(tf.nn.softmax(y_prediction_batch_without_softmax), 1)
        y_output = tf.nn.softmax(y_prediction_batch_without_softmax, name="y_output")

        ce = tl.cost.cross_entropy(y_prediction_batch_without_softmax, y_batch, name='cost')

        """ 需给后面的全连接层引入L2 normalization，惩罚模型的复杂度，避免overfitting """
        # L2 for the MLP, without this, the accuracy will be reduced by 15%.
        L2 = 0
        for p in tl.layers.get_variables_with_name('relu/W', True, True):
            L2 += tf.contrib.layers.l2_regularizer(0.004)(p)
        # 加上L2模型复杂度惩罚项后，得到最终真正的cost
        cost = ce + L2

        correct_prediction = tf.equal(tf.cast(tf.argmax(y_prediction_batch_without_softmax, 1), y_TF_DTYPE), y_batch)
        # correct_prediction = tf.Print(correct_prediction, [correct_prediction], "correct_prediction: ")
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return net, cost, accuracy, y_output


# Train, validate and test
batch_size = 128
n_epoch = 16
n_step_per_epoch = int(len(train_set[y]) / batch_size)
n_step = n_epoch * n_step_per_epoch
print_freq = 1
checkpoint_freq = 4
learning_rate = 0.0001

model_ckpt_file_name = os.path.join(working_directory, "checkpoint", "model-quickdraw-cnn.ckpt")
resume = True  # load model, resume from previous checkpoint?


def distort_fn(X, is_train=False):
    # print('begin', X.shape, np.min(X), np.max(X))

    if is_train == True:
        # 1. Randomly flip the image horizontally.
        X = tf.image.random_flip_left_right(X)

    # X = tf.image.per_image_standardization(X)

    # print('after norm', X.shape, np.min(X), np.max(X), np.mean(X))
    return X


def save_model():
    model_type = "saved-model"
    latest_model_directory = f'{model_type}-{time.strftime("%Y%m%d%H%M%S", time.localtime())}'
    saved_model_directory = os.path.join(working_directory, latest_model_directory)
    if not os.path.exists(saved_model_directory):
        tf.saved_model.simple_save(session, saved_model_directory,
                                   inputs={"X": X_batch_ph},
                                   outputs={"y_output": y_prediction_})
        dist_directory = os.path.join(".", model_type)
        if os.path.exists(dist_directory):
            os.remove(dist_directory)
        os.symlink(saved_model_directory, dist_directory, target_is_directory=True)


with tf.device('/cpu:0'):
    session = tf.Session(config=config)

    #
    # Connect to tfdbg dashboard by ```http://localhost:6006#debugger```
    # when the following command is issued.
    #
    # ```bash
    # $ tensorboard --logdir logs --port 6006 --debugger_port 6064
    # ```
    #
    # session = tfdebug.TensorBoardDebugWrapperSession(session, "albert-mbp.local:6064")

    X_batch_ph = tf.placeholder(dtype=X_TF_DTYPE, shape=[None, image_height, image_width, image_depth], name='X_batch')
    y_batch_ph = tf.placeholder(dtype=y_TF_DTYPE, shape=[None], name='y_batch')
    # X_batch_ph = tf.placeholder(dtype=X_TF_DTYPE, shape=[batch_size, image_height, image_width, image_depth], name='X')
    # y_batch_ph = tf.placeholder(dtype=y_TF_DTYPE, shape=[batch_size], name='y')

    def perform_minibatch(run_list, X, y, batch_size, is_train=False):
        n_batch, sum_loss, sum_accuracy = 0, 0, 0
        for X_batch_a, y_batch_a in tl.iterate.minibatches(X, y, batch_size, shuffle=is_train):
            # data augmentation for training
            # X_batch_a = tl.prepro.threading_data(X_batch_a, fn=distort_fn, is_train=is_train)

            cost, accuracy = 0, 0
            if is_train:
                _, cost, accuracy = session.run(
                    run_list, feed_dict={X_batch_ph: X_batch_a, y_batch_ph: y_batch_a}
                )
            else:
                cost, accuracy = session.run(
                    run_list, feed_dict={X_batch_ph: X_batch_a, y_batch_ph: y_batch_a}
                )

            sum_loss += cost
            sum_accuracy += accuracy
            n_batch += 1
        return n_batch, sum_loss, sum_accuracy


    with tf.device('/gpu:0'):  # <-- remove it if you don't have GPU
        # Build the model
        print("### Train Network model ###")
        network_, cost_, accuracy_, y_prediction_ = model_batch_normalization(
            X_batch_ph, y_batch_ph, n_category, reuse=None, is_train=True
        )
        print("### Reuse this Train Network model for validation and test ###")
        _, cost_test_, accuracy_test_, y_prediction_test_ = model_batch_normalization(
            X_batch_ph, y_batch_ph, n_category, reuse=True, is_train=False
        )

    # Define the training optimizer
    with tf.device('/gpu:0'):  # <-- remove it if you don't have GPU
        train_op_ = tf.train.AdamOptimizer(learning_rate).minimize(cost_)

    tl.layers.initialize_global_variables(session)

    # Attach the graph for TensorBoard writer
    # tfboard_file_writer.add_graph(tf.get_default_graph())
    tfboard_file_writer.add_graph(session.graph)

    if resume and os.path.isfile(model_ckpt_file_name):
        print("Load existing model " + "!" * 10)
        saver = tf.train.Saver()
        saver.restore(session, model_ckpt_file_name)

    print("### Network parameters ###")
    network_.print_params(False)
    print("### Network layers ###")
    network_.print_layers()

    print('   learning_rate: %f' % learning_rate)
    print('   batch_size: %d' % batch_size)
    print('   n_epoch: %d, step in an epoch: %d, total n_step: %d' % (n_epoch, n_step_per_epoch, n_step))

    step, sum_batch, sum_loss, sum_accuracy = 0, 0, 0, 0
    for epoch in range(n_epoch):
        start_time = time.time()

        n_batch_a_epoch, cost_a_epoch, accuracy_a_epoch = perform_minibatch(
            [train_op_, cost_, accuracy_],
            train_set[X], train_set[y], batch_size, is_train=True
        )
        sum_batch += n_batch_a_epoch
        sum_loss += cost_a_epoch
        sum_accuracy += accuracy_a_epoch
        step += n_batch_a_epoch

        assert n_batch_a_epoch == n_step_per_epoch

        if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
            print("Epoch %d : Step %d-%d of %d took %fs" %
                  (epoch + 1, step - n_step_per_epoch, step, n_step, time.time() - start_time))
            print("   train loss: %f" % (sum_loss / sum_batch))
            print("   train accuracy: %f" % (sum_accuracy / sum_batch))
            sum_batch, sum_loss, sum_accuracy = 0, 0, 0

            n_batch_a_epoch, cost_a_epoch, accuracy_a_epoch = perform_minibatch(
                [cost_test_, accuracy_test_],
                validation_set[X], validation_set[y], batch_size
            )
            print("   validation loss: %f" % (cost_a_epoch / n_batch_a_epoch))
            print("   validation accuracy: %f" % (accuracy_a_epoch / n_batch_a_epoch))

            n_batch_a_epoch, cost_a_epoch, accuracy_a_epoch = perform_minibatch(
                [cost_test_, accuracy_test_],
                test_set[X], test_set[y], batch_size
            )
            print("   test loss: %f" % (cost_a_epoch / n_batch_a_epoch))
            print("   test accuracy: %f" % (accuracy_a_epoch / n_batch_a_epoch))

        # Save model when checkpoint
        if (epoch + 1) % checkpoint_freq == 0:
            print("Saving checkpoint... " + "!" * 10)
            saver = tf.train.Saver()
            save_path = saver.save(session, model_ckpt_file_name)
            print("Saving model... " + "!" * 10)
            save_model()

    save_model()

    tfboard_file_writer.flush()
    tfboard_file_writer.close()

    session.close()

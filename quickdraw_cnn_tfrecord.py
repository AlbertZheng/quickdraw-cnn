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
categories_filename = 'categories.txt'
categories_file_url_source = 'https://raw.githubusercontent.com/googlecreativelab/quickdraw-dataset/master/'
# curl -v --socks5 127.0.0.1:1080 "https://storage.cloud.google.com/quickdraw_dataset/full/numpy_bitmap/aircraft%20carrier.npy"
# npy_dataset_url_source = 'https://storage.cloud.google.com/quickdraw_dataset/full/numpy_bitmap/'
npy_dataset_url_source = 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/'

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

n_category = 345  # The maximum of category number is up to 345


def print_dataset_shape(X_name, X, y_name, y):
    print(X_name + '.shape ', X.shape, end='\t,\t')
    print(y_name + '.shape ', y.shape)
    print('%s.dtype %s\t,\t%s.dtype %s' % (X_name, X.dtype, y_name, y.dtype))


def show_image(X, y, categories):
    plt.imshow(X.reshape(image_height, image_width), cmap="gray", interpolation='nearest')
    plt.title(categories[y])
    plt.show()


def load_quickdraw_dataset(
    n_category=10, n_train_example_per_category=50000, plotable=False
):
    """ Download the quick draw data set. """
    n_validation_example_per_category = int(n_train_example_per_category / 0.7 * 0.2)
    n_test_set_example_category = int(n_train_example_per_category / 0.7 * 0.1)

    # Download the categories file
    tl.files.utils.maybe_download_and_extract(categories_filename, dataset_directory, categories_file_url_source)

    tl.logging.info("Load or Download quick draw > {}".format(dataset_directory))

    train_set = {X: np.empty([0, image_size], dtype=X_NUMPY_DTYPE), y: np.empty([0], dtype=y_NUMPY_DTYPE)}
    validation_set = {X: np.empty([0, image_size], dtype=X_NUMPY_DTYPE), y: np.empty([0], dtype=y_NUMPY_DTYPE)}
    test_set = {X: np.empty([0, image_size], dtype=X_NUMPY_DTYPE), y: np.empty([0], dtype=y_NUMPY_DTYPE)}

    categories = [line.rstrip('\n') for line in open(f"{dataset_directory}/{categories_filename}")]
    # categories = [line.rstrip('\n') for line in open(os.path.join(working_directory, categories_filename))]
    for category_index, category_name in enumerate(categories):
        if category_index == n_category:
            break

        # filename = category_name.replace(' ', '%20') + '.npy'
        filename = urllib.parse.quote(category_name) + '.npy'
        # filename = category_name + '.npy'
        tl.files.utils.maybe_download_and_extract(filename, dataset_directory, npy_dataset_url_source)

        data = np.load(os.path.join(dataset_directory, filename))
        size_per_category = data.shape[0]
        labels = np.full(size_per_category, category_index)

        print(f"### Category '{category_name}' dataset info ###")
        print_dataset_shape("data", data, "labels", labels)

        # Randomize the dataset
        permutation = np.random.permutation(size_per_category)
        data = data[permutation, :]
        labels = labels[permutation]

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
        number_end += n_test_set_example_category
        # test_set[X] = np.concatenate((test_set[X], data[number_begin:number_end, :]), axis=0)
        test_set[X] = np.vstack((test_set[X], data[number_begin:number_end, :]))
        test_set[y] = np.append(test_set[y], labels[number_begin:number_end])

        print_dataset_shape("train_set[X]", train_set[X], "train_set[y]", train_set[y])
        print_dataset_shape("validation_set[X]", validation_set[X], "validation_set[y]", validation_set[y])
        print_dataset_shape("test_set[X]", test_set[X], "test_set[y]", test_set[y])

    # Reshape for CNN input
    train_set[X] = train_set[X].reshape(input_layer_X_shape_batch)
    validation_set[X] = validation_set[X].reshape(input_layer_X_shape_batch)
    test_set[X] = test_set[X].reshape(input_layer_X_shape_batch)

    """
    if plotable:
        r = randint(0, n_category * n_train_set_per_category - 1)
        show_image(train_set[X][r], train_set[y][r], categories)
    
        r = randint(0, n_category * n_validation_example_per_category - 1)
        show_image(validation_set[X][r], validation_set[y][r], categories)
    
        r = randint(0, n_category * n_test_set_example_category - 1)
        show_image(test_set[X][r], test_set[y][r], categories)
    """

    return categories, train_set, validation_set, test_set


def data_to_tfrecord(images, labels, filename):
    """ Save data set into TFRecord. """
    filename = os.path.join(working_directory, filename)

    """
    if os.path.isfile(filename):
        print("%s exists" % filename)
        return
    """

    print("Current directory: %s " % os.getcwd())
    print("Converting data into %s ..." % filename)

    writer = tf.python_io.TFRecordWriter(filename)
    for index, img in enumerate(images):
        label = int(labels[index])
        img_raw = img.tobytes()

        # Visualize a image
        if index <= 0:
            """
            tl.visualize.frame(np.asarray(img, dtype=np.uint8), second=1,
                               saveable=False, name='label: {}: {}'.format(label, categories[label]), fig_idx=1236)
            """
            # Just for debug
            print(f"### Debug data_to_tfrecord() for '{filename}' ###")
            print("category id: %d" % label)
            print("image.shape: \n", img.shape)
            print("len(img_raw): \n", len(img_raw))
            # print("img_raw: \n", np.fromstring(img_raw, X_NUMPY_DTYPE))

        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                    'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                }
            )
        )
        writer.write(example.SerializeToString())  # Serialize To String
    writer.close()


def read_and_decode(filename, is_train=None):
    """ Return tensor to read from TFRecord. """
    filename = os.path.join(working_directory, filename)

    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'img_raw': tf.FixedLenFeature([], tf.string),
        }
    )

    img = tf.decode_raw(features['img_raw'], X_TF_DTYPE)
    img = tf.reshape(img, input_layer_X_shape)

    # You can do more image distortion here for training data
    """
    if is_train == True:
        # 1. Randomly flip the image horizontally.
        img = tf.image.random_flip_left_right(img)

    img = tf.image.per_image_standardization(img)
    """

    label = tf.cast(features['label'], y_TF_DTYPE)

    return img, label


def model_batch_normalization(X_batch, y_batch, output_units, reuse, is_train):
    """ Define the network model """
    W_init1 = tf.truncated_normal_initializer(stddev=5e-2)
    W_init2 = tf.truncated_normal_initializer(stddev=0.04)
    bias_init = tf.constant_initializer(value=0.1)

    with tf.variable_scope("model", reuse=reuse):
        net = InputLayer(X_batch, name='input')
        net = Conv2d(net, 64, (5, 5), (1, 1), padding='SAME',
                     W_init=W_init1, b_init=None, name='cnn1')
        net = BatchNormLayer(net, is_train, act=tf.nn.relu, name='batch1')
        net = MaxPool2d(net, (3, 3), (2, 2), padding='SAME', name='pool1')

        net = Conv2d(net, 64, (5, 5), (1, 1), padding='SAME',
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

        # For inference using this model
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


# Open TensorBoard logs writer
tfboard_file_writer = tf.summary.FileWriter('logs')

# Download data
categories, train_set, validation_set, test_set = load_quickdraw_dataset(n_category, 2000)

# Save data into TFRecord files
train_set_tfrecord_filename = "train-quickdraw.tfrecord"
validation_set_tfrecord_filename = "validation-quickdraw.tfrecord"
test_set_tfrecord_filename = "test-quickdraw.tfrecord"

data_to_tfrecord(images=train_set[X], labels=train_set[y], filename=train_set_tfrecord_filename)
data_to_tfrecord(images=validation_set[X], labels=validation_set[y], filename=validation_set_tfrecord_filename)
data_to_tfrecord(images=test_set[X], labels=test_set[y], filename=test_set_tfrecord_filename)

# Train, validate and test
batch_size = 128
n_epoch = 100
n_step_epoch = int(len(train_set[y]) / batch_size)
n_step = n_epoch * n_step_epoch
print_freq = 1
learning_rate = 0.0001

model_ckpt_file_name = os.path.join(working_directory, "checkpoint", "model-quickdraw-cnn.ckpt")
resume = True  # load model, resume from previous checkpoint?


with tf.device('/cpu:0'):
    session = tf.Session(config=config)

    # Connect to tfdbg dashboard by ```http://localhost:6006#debugger```
    # when the following command is issued.
    #
    # ```bash
    # $ tensorboard --logdir logs --port 6006 --debugger_port 6064
    # ```
    # session = tfdebug.TensorBoardDebugWrapperSession(session, "albert-mbp.local:6064")

    """ Prepare data in cpu """
    # The reader tensor for a single data Example
    X_train_single, y_train_single = read_and_decode(train_set_tfrecord_filename, True)
    X_validation_single, y_validation_single = read_and_decode(validation_set_tfrecord_filename, True)
    X_test_single, y_test_single = read_and_decode(test_set_tfrecord_filename, False)

    # The input streaming tensor for a batch augmented data per a training step by using multi-threads.
    X_train_batch, y_train_batch = tf.train.shuffle_batch(
        [X_train_single, y_train_single], batch_size=batch_size,
        capacity=2000, min_after_dequeue=1000, num_threads=32
        # set the number of threads here
    )

    # For testing, uses batch() instead of shuffle_batch()
    X_test_batch, y_test_batch = tf.train.batch(
        [X_test_single, y_test_single], batch_size=batch_size, capacity=50000, num_threads=32
    )

    with tf.device('/gpu:0'):  # <-- remove it if you don't have GPU
        # Uses batch normalization
        print("### Train Network model ###")
        network_, cost_, accuracy_, y_prediction_ = model_batch_normalization(
            X_train_batch, y_train_batch, n_category, reuse=None, is_train=True
        )
        print("### Reuse this Train Network model ###")
        _, cost_test_, accuracy_test_, y_prediction_test_ = model_batch_normalization(
            X_test_batch, y_test_batch, n_category, reuse=True, is_train=False
        )

    # Define the training optimizer
    with tf.device('/gpu:0'):  # <-- remove it if you don't have GPU
        train_op_ = tf.train.AdamOptimizer(learning_rate).minimize(cost_)

    tl.layers.initialize_global_variables(session)

    # Attach the graph for TensorBoard writer
    #tfboard_file_writer.add_graph(tf.get_default_graph())
    tfboard_file_writer.add_graph(session.graph)

    if resume and os.path.isfile(model_ckpt_file_name):
        print("Load existing model " + "!" * 10)
        saver = tf.train.Saver()
        saver.restore(session, model_ckpt_file_name)

    print("### Train Network parameters ###")
    network_.print_params(False)
    print("### Train Network layers ###")
    network_.print_layers()

    print('   learning_rate: %f' % learning_rate)
    print('   batch_size: %d' % batch_size)
    print('   n_epoch: %d, step in an epoch: %d, total n_step: %d' % (n_epoch, n_step_epoch, n_step))

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=session, coord=coord)
    step = 0
    for epoch in range(n_epoch):
        start_time = time.time()
        sum_loss, sum_accuracy, n_batch = 0, 0, 0
        for s in range(n_step_epoch):
            cost_a_batch, accuracy_a_batch, y_prediction_a_batch, _ = session.run(
                [cost_, accuracy_, y_prediction_, train_op_]
            )
            step += 1
            sum_loss += cost_a_batch
            sum_accuracy += accuracy_a_batch
            n_batch += 1
            # print("training prediction: ", y_prediction_a_batch)

        if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
            print("Epoch %d : Step %d-%d of %d took %fs" %
                  (epoch, step - n_step_epoch, step, n_step, time.time() - start_time))
            print("   train loss: %f" % (sum_loss / n_batch))
            print("   train accuracy: %f" % (sum_accuracy / n_batch))

            sum_loss, sum_accuracy, n_batch = 0, 0, 0
            for _ in range(int(len(test_set[y]) / batch_size)):
                cost_a_batch, accuracy_a_batch = session.run([cost_test_, accuracy_test_])
                sum_loss += cost_a_batch
                sum_accuracy += accuracy_a_batch
                n_batch += 1
            print("   test loss: %f" % (sum_loss / n_batch))
            print("   test accuracy: %f" % (sum_accuracy / n_batch))

        # Save model when checkpoint
        if (epoch + 1) % (print_freq * 50) == 0:
            print("Save model " + "!" * 10)
            saver = tf.train.Saver()
            save_path = saver.save(session, model_ckpt_file_name)
            # You can also save model into npz
            # tl.files.save_npz(network.all_params, name='model-quickdraw-cnn.npz', sess=session)

    model_type = "saved-model"
    latest_model_directory = f'{model_type}-{time.strftime("%Y%m%d%H%M%S", time.localtime())}'
    saved_model_directory = os.path.join(working_directory, latest_model_directory)
    tf.saved_model.simple_save(session, saved_model_directory,
                               inputs={"X": X_test_batch},
                               outputs={"y_output": y_prediction_test_})
    dist_directory = os.path.join(".", model_type)
    if os.path.exists(dist_directory):
        os.remove(dist_directory)
    os.symlink(saved_model_directory, dist_directory, target_is_directory=True)

    tfboard_file_writer.flush()
    tfboard_file_writer.close()

    coord.request_stop()
    coord.join(threads)
    session.close()

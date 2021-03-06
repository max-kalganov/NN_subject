'''Trains a simple convnet on the MNIST dataset and embeds test data.
The test data is embedded using the weights of the final dense layer, just
before the classification head. This embedding can then be visualized using
TensorBoard's Embedding Projector.
'''

from __future__ import print_function

from os import makedirs
from os.path import exists, join

import tensorflow as tf
# import keras
# from keras.callbacks import TensorBoard
# from keras.datasets import mnist
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
# from keras import backend as K

import numpy as np

if __name__ == '__main__':
    batch_size = 128
    num_classes = 10
    epochs = 12
    log_dir = './logs'

    if not exists(log_dir):
        makedirs(log_dir)

    # input image dimensions
    img_rows, img_cols = 28, 28

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    if tf.keras.backend.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # save class labels to disk to color data points in TensorBoard accordingly
    with open(join(log_dir, 'metadata.tsv'), 'w') as f:
        np.savetxt(f, y_test)

    # convert class vectors to binary class matrices
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    tensorboard = tf.keras.callbacks.TensorBoard(batch_size=batch_size,
                              embeddings_freq=1,
                              embeddings_layer_names=['features'],
                              embeddings_metadata='metadata.tsv',
                              embeddings_data=x_test)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(128, activation='relu', name='features'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer=tf.keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              callbacks=[tensorboard],
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # You can now launch tensorboard with `tensorboard --logdir=./logs` on your
    # command line and then go to http://localhost:6006/#projector to view the
    # embeddings
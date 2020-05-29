from os.path import join

import pandas as pd
from tensorflow.keras.layers import Activation
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from matplotlib import pyplot as plt
import numpy as np
# import TensorBoard as tb
from tensorboard.program import TensorBoard

from data_generation import get_input_timeseries


def custom_activation(x):
    return 1 / x


get_custom_objects().update({'custom_activation': Activation(custom_activation)})


class Classifier:
    def __init__(self):
        self.classif = Sequential()

        self.classif.add(Dense(20,
                               activation='relu',
                               kernel_initializer='random_normal',
                               input_dim=1))
        # self.classif.add(Activation(custom_activation, name='SpecialActivation'))

        self.classif.add(Dense(20,
                               activation='relu',
                               kernel_initializer='random_normal',
                               input_dim=20))

        self.classif.add(Dense(20,
                               activation='sigmoid',
                               kernel_initializer='random_normal',
                               input_dim=20))

        self.classif.add(Dense(1,
                               activation='sigmoid',
                               kernel_initializer='random_normal',
                               input_dim=20))

        self.classif.compile(optimizer='Nadam', loss=MeanSquaredError(), metrics=["mean_squared_error"])

    def train(self, x_train, y_train):
        batch_size = 1000

        self.classif.fit(x_train,
                         y_train,
                         batch_size=batch_size,
                         epochs=5000,
                         shuffle=True)
        return self.classif.evaluate(x_train, y_train)

    def test(self, x_test):
        y_pred = self.classif.predict(x_test)
        return y_pred

    def save(self):
        self.classif.save('data/classifier.h5')
        print("classifier is saved")

    def load(self, classifier_name: str = 'classifier'):
        self.classif = load_model(f'data/{classifier_name}.h5')
        print("classifier is loaded")


def test(classif: Classifier, x_test, y_test):
    y_pred = classif.test(x_test)
    y_pred = y_pred.reshape((y_pred.shape[0],))
    print(f"mse on test = {np.mean((y_test - y_pred)**2)}")


def get_full_dataset():
    y = get_input_timeseries(100)
    t = np.arange(100 * 100 + 1)
    return t, y


def get_dataset():
    y = get_input_timeseries(100)
    t = np.arange(100*100 + 1)
    train_size = int(0.7 * y.shape[0])
    indices = np.random.choice(t.shape[0], train_size)
    x_train, y_train = t[indices], y[indices]
    x_test = t[np.invert(np.isin(t, indices))]
    y_test = y[x_test]

    return (x_train, y_train), (x_test, y_test)
    # return (x_train.reshape(x_train.shape[0], 1), y_train.reshape(y_train.shape[0], 1)), \
    #        (x_test.reshape(x_test.shape[0], 1), y_test.reshape(y_test.shape[0], 1))


def train_and_test():
    (x_train, y_train), (x_test, y_test) = get_dataset()
    # x_train = np.arange(1000).astype(float)
    # y_train = 1/x_train
    # x_test = np.arange()

    classif = Classifier()
    loss, mse = classif.train(x_train, y_train)

    print(f"\ntraining results for dataset:\nloss = {loss}\nmse = {mse}\n")
    classif.save()

    test(classif, x_test, y_test)


def load_and_test():
    (x_train, y_train), (x_test, y_test) = get_dataset()

    classif = Classifier()
    classif.load()

    test(classif, x_test, x_train)


def load_and_show():
    x, y = get_full_dataset()
    cl = Classifier()
    cl.load()
    y_pred = cl.test(x)

    plt.plot(y)
    plt.plot(y_pred)
    plt.show()


if __name__ == '__main__':
    load_and_show()
    # load_and_test()
    # train_and_test()

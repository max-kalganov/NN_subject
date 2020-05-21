from os.path import join

import pandas as pd
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import numpy as np
# import TensorBoard as tb
from tensorboard.program import TensorBoard

from data_generation import get_input_timeseries


class Classifier:
    def __init__(self):
        self.classif = Sequential()

        self.classif.add(Dense(20,
                               activation='sigmoid',
                               kernel_initializer='random_normal',
                               input_dim=1))

        self.classif.add(Dense(20,
                               activation='sigmoid',
                               kernel_initializer='random_normal',
                               input_dim=20))

        self.classif.add(Dense(20,
                               activation='sigmoid',
                               kernel_initializer='random_normal',
                               input_dim=20))

        self.classif.add(Dense(1,
                               activation='relu',
                               kernel_initializer='random_normal',
                               input_dim=20))

        self.classif.compile(optimizer='adam', loss=MeanSquaredError(), metrics=['accuracy'])

    def train(self, x_train, y_train):
        batch_size = 10

        self.classif.fit(x_train,
                         y_train,
                         batch_size=batch_size,
                         epochs=100,
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
    cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
    t_pos = cm[0, 0]
    t_neg = cm[1, 1]
    f_pos = cm[0, 1]
    f_neg = cm[1, 0]
    test_acc = (t_pos + t_neg)/(t_pos + t_neg + f_pos + f_neg)
    print(f"test accuracy = {test_acc}")


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
    classif = Classifier()
    loss, acc = classif.train(x_train, y_train)

    print(f"\ntraining results for dataset:\nloss = {loss}\naccuracy = {acc}\n")
    classif.save()

    test(classif, x_test, y_test)


if __name__ == '__main__':
    train_and_test()

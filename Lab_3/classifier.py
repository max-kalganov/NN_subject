from os.path import join

import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import numpy as np
from utils import get_dataset
# import TensorBoard as tb
from tensorboard.program import TensorBoard


class BinClassifier:
    def __init__(self):
        self.classif = Sequential()

        self.classif.add(Dense(28*28,
                               activation='relu',
                               kernel_initializer='random_normal',
                               input_dim=28 * 28,
                               name='features1'))

        self.classif.add(Dense(10,
                               activation='sigmoid',
                               kernel_initializer='random_normal',
                               input_dim=28*28,
                               name='features'))

        self.classif.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def train(self, x_train, y_train):
        batch_size = 1000

        self.classif.fit(x_train,
                         y_train,
                         batch_size=batch_size,
                         epochs=50,
                         shuffle=True)
        return self.classif.evaluate(x_train, y_train)

    def test(self, x_test, full_return: bool = False):
        y_pred = self.classif.predict(x_test)
        return y_pred if full_return else np.maximum(y_pred-0.5, 0)

    def save(self):
        self.classif.save('data/classifier.h5')
        print("classifier is saved")

    def load(self):
        self.classif = load_model('data/classifier.h5')
        print("classifier is loaded")


def test(binclassif: BinClassifier, x_test, y_test):
    y_pred = binclassif.test(x_test)
    cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
    t_pos = cm[0, 0]
    t_neg = cm[1, 1]
    f_pos = cm[0, 1]
    f_neg = cm[1, 0]
    test_acc = (t_pos + t_neg)/(t_pos + t_neg + f_pos + f_neg)
    print(f"test accuracy = {test_acc}")


def train_and_test():
    (x_train, y_train), (x_test, y_test) = get_dataset()
    binclassif = BinClassifier()
    loss, acc = binclassif.train(x_train, y_train)

    print(f"\ntraining results for dataset:\nloss = {loss}\naccuracy = {acc}\n")
    binclassif.save()

    test(binclassif, x_test, y_test)


if __name__ == '__main__':
    train_and_test()

import pandas as pd
from sklearn.model_selection import train_test_split
from ct import get_ds_path, X_COORD, Y_COORD, CLASSES
from keras import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt


class BinClassifier:
    def __init__(self, one_layer: bool):
        self.classif = Sequential()

        input_dim = 2
        if not one_layer:
            self.classif.add(Dense(3,
                                   activation='relu',
                                   kernel_initializer='random_normal',
                                   input_dim=input_dim))
            input_dim = 3

        self.classif.add(Dense(1,
                               activation='sigmoid',
                               kernel_initializer='random_normal',
                               input_dim=input_dim))

        self.classif.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def train(self, x_train, y_train):
        self.classif.fit(x_train, y_train, batch_size=10, epochs=200)
        return self.classif.evaluate(x_train, y_train)

    def test(self, x_test):
        y_pred = self.classif.predict(x_test)
        return y_pred > 0.5


def train_and_test(one_layer: bool, name: str):
    def lin_func(x):
        w1 = binclassif.classif.layers[0].get_weights()[0][0, 0]
        w2 = binclassif.classif.layers[0].get_weights()[0][1, 0]
        w3 = binclassif.classif.layers[0].get_weights()[1][0]
        return (-w1 * x - w3) / w2

    df = pd.read_csv(get_ds_path(name)).sample(frac=1).drop('Unnamed: 0', axis=1)
    X = df[[X_COORD, Y_COORD]].to_numpy()
    Y = df[CLASSES].to_numpy()
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
    binclassif = BinClassifier(one_layer=one_layer)
    loss, acc = binclassif.train(x_train, y_train)
    print(f"\ntraining results for dataset - {name}:\nloss = {loss}\naccuracy = {acc}\n")
    y_pred = binclassif.test(x_test)
    cm = confusion_matrix(y_test, y_pred)
    t_pos = cm[0, 0]
    t_neg = cm[1, 1]
    f_pos = cm[0, 1]
    f_neg = cm[1, 0]
    test_acc = (t_pos + t_neg)/(t_pos + t_neg + f_pos + f_neg)
    print(f"test accuracy = {test_acc}")

    if one_layer:
        cl1 = X[Y == 0]
        cl2 = X[Y == 1]
        plt.plot(cl1[:, 0], cl1[:, 1], "bo")
        plt.plot(cl2[:, 0], cl2[:, 1], "ro")
        x1 = [-10, 20]
        y1 = [lin_func(x1[0]), lin_func(x1[1])]
        plt.plot(x1, y1)
        plt.show()


def train_and_test_classifiers():
    train_and_test(one_layer=True,
                   name="ds1")
    train_and_test(one_layer=True,
                   name="ds2")
    train_and_test(one_layer=True,
                   name="ds3")
    train_and_test(one_layer=False,
                   name="ds4")
    train_and_test(one_layer=False,
                   name="ds5")
    train_and_test(one_layer=False,
                   name="ds6")


if __name__ == '__main__':
    train_and_test_classifiers()

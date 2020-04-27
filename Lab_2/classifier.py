import pandas as pd
from sklearn.model_selection import train_test_split
from ct import get_ds_path, X_COORD, Y_COORD, CLASSES
from keras import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix


class BinClassifier:
    def __init__(self, one_layer: bool):
        self.classif = Sequential()

        if not one_layer:
            self.classif.add(Dense(3, activation='relu', kernel_initializer='random_normal', input_dim=2))

        self.classif.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))

        self.classif.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def train(self, x_train, y_train):
        self.classif.fit(x_train, y_train, batch_size=10, epochs=10)
        return self.classif.evaluate(x_train, y_train)

    def test(self, x_test):
        y_pred = self.classif.predict(x_test)
        return y_pred > 0.5


def train_and_test(one_layer: bool, name: str):
    df = pd.read_csv(get_ds_path(name)).sample(frac=1).drop('Unnamed: 0')
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
    return binclassif.classif.weights


def train_and_test_classifiers():
    w1 = train_and_test(one_layer=True,
                        name="ds1")
    w2 = train_and_test(one_layer=True,
                        name="ds2")
    w3 = train_and_test(one_layer=True,
                        name="ds3")
    w4 = train_and_test(one_layer=False,
                        name="ds4")
    w5 = train_and_test(one_layer=False,
                        name="ds5")
    w6 = train_and_test(one_layer=False,
                        name="ds6")


if __name__ == '__main__':
    train_and_test(one_layer=True, name="ds1")

from classifier import BinClassifier, test
from utils import get_dataset, EXAMPLE_NAMES, EXAMPLE_ANSWERS
from skimage.io import imread
import numpy as np


def test_all_local(bc: BinClassifier, full_res: bool):
    correct = 0
    for name, res in zip(EXAMPLE_NAMES, EXAMPLE_ANSWERS):
        pred, cur_res = test_local(bc, name, full_res)
        if full_res:
            print(f"prediction = {pred}")
        print(f"current result = {cur_res}, real answer = {res} --- {cur_res == res} filename = {name}")
        correct += (res == cur_res)
    print(f"num of correct answers = {correct}")
    print(f"all number of examples = {len(EXAMPLE_NAMES)}")


def test_local(bc: BinClassifier, pict_name: str, full_res: bool):
    image = imread(fname=f'data/{pict_name}.png', as_gray=True)
    # image = image*255
    pred = bc.test(image.reshape(1, image.shape[0] * image.shape[1]), full_return=full_res)
    res = np.argmax(pred)
    return pred, res


def check_classifier(with_external_picture: bool):
    bc = BinClassifier()
    bc.load(classifier_name="classifier_without_examples")

    if not with_external_picture:
        (x_train, y_train), (x_test, y_test) = get_dataset()
        test(bc, x_test, y_test)
    else:
        test_all_local(bc, full_res=False)


if __name__ == '__main__':
    check_classifier(with_external_picture=True)

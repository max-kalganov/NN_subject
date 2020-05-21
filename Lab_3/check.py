from classifier import BinClassifier, test
from utils import get_dataset
from skimage.io import imread
import numpy as np


def test_local(bc: BinClassifier, pict_name: str, full_res: bool):
    image = imread(fname=f'data/{pict_name}.png', as_gray=True)
    # image = image*255
    pred = bc.test(image.reshape(1, image.shape[0] * image.shape[1]), full_return=full_res)
    res = np.argmax(pred)
    print(f"number - {res}")
    print(f"pred = {pred}")


def check_classifier(with_external_picture: bool):
    bc = BinClassifier()
    bc.load()

    if not with_external_picture:
        (x_train, y_train), (x_test, y_test) = get_dataset()
        test(bc, x_test, y_test)
    else:
        test_local(bc, pict_name="test_pict1", full_res=False)


if __name__ == '__main__':
    check_classifier(with_external_picture=False)

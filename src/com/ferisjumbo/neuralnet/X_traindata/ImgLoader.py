import cv2
import numpy as np

img_dir = (
    "src/com/ferisjumbo/neuralnet/X_traindata/"  # The directory to the training data
)


def get_img_array(fileloc):
    """
    Gets the specified image from this folder and returns a single array of numbers from 0-255
    based on the grayscale.

    Args:
        fileloc (string): src/com/ferisjumbo/neuralnet/X_train-data + fileloc

    Returns:
        array[]: 1,024 numbers to be exact
    """
    image = cv2.imread(img_dir + fileloc)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    data = []
    for x in img:
        for y in x:
            data.append(y)
    return data

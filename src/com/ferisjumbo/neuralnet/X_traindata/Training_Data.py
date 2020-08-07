import numpy as np
import X_traindata.ImgLoader as imgL
import NetworkUtils as nU

numbers = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
]


def get_training_data():
    total_datasets = 1

    X = []
    y = []

    for dataset in range(total_datasets):
        for number in numbers:
            fileloc = number + str(dataset) + ".png"
            X.append(nU.normalize_data(imgL.get_img_array(fileloc)))
            y.append(numbers.index(number))

    return X, y


def get_testing_data(number, dataset):
    X = []
    y = []

    fileloc = number + str(dataset) + ".png"
    X.append(nU.normalize_data(imgL.get_img_array(fileloc)))
    y.append(numbers.index(number))

    return X, y

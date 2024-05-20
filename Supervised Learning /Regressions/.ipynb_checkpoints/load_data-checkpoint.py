import numpy as np


def load_model(feature: int = 0):
    with open("./../Dataset_custom/houses.txt") as file:
        x, y = [], []
        for i in file:
            i = list(map(float, i.split(sep=',')))
            x.append(i[:feature])
            y.append(i[-1])
        x, y = np.array(x), np.array(y)
    return x, y
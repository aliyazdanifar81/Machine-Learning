import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class LoadModel:
    def __init__(self):
        self.paths = {
            "house": "/home/ali/Desktop/python/Machine Learning/Dataset_custom/houses.txt",
            "Titanic_train": "/home/ali/Desktop/python/Machine Learning/Dataset_custom/Titanic/train.csv",
            "Titanic_test": "/home/ali/Desktop/python/Machine Learning/Dataset_custom/Titanic/test.csv",
            "Real_estate": "/home/ali/Desktop/python/Machine Learning/Dataset_custom/Real estate.csv",
            "ex2data2": "/home/ali/Desktop/python/Machine Learning/Dataset_custom/ex2data2.txt",
            "ex2data1": "/home/ali/Desktop/python/Machine Learning/Dataset_custom/ex2data1.txt",
            "Cardio": "/home/ali/Desktop/python/Machine Learning/Dataset_custom/Cardio/cardio_train.csv",
            "hpart_train": "/home/ali/Desktop/python/Machine Learning/Dataset_custom/house-prices-advanced-regression-techniques/train.csv",
            # hpart stands for House Prices - Advanced Regression Techniques
            "hpart_test": "/home/ali/Desktop/python/Machine Learning/Dataset_custom/house-prices-advanced-regression-techniques/test.csv"
            # hpart stands for House Prices - Advanced Regression Techniques
        }

    def load_house(self, feature: int = 0):
        with open(self.paths["house"]) as file:
            x, y = [], []
            for i in file:
                i = list(map(float, i.split(sep=',')))
                x.append(i[:feature])
                y.append(i[-1])
            x, y = np.array(x), np.array(y)
        return x, y

    def load_titanic(self):
        train, test = pd.read_csv(self.paths["Titanic_train"]), pd.read_csv(self.paths["Titanic_test"])
        return train, test

    def load_realestate(self):
        df = pd.read_csv(self.paths["Real_estate"])
        x = df.drop(["No", "Y house price of unit area"], axis=1)
        y = df["Y house price of unit area"]
        return x, y

    def load_ex2data2(self):
        with open(self.paths["ex2data2"], 'r') as file:
            x, y = [], []
            for i in file:
                i = list(map(float, i.split(sep=',')))
                x.append(i[:2])
                y.append(i[-1])
            x, y = np.array(x), np.array(y)
        return x, y

    def load_ex2data1(self):
        with open(self.paths["ex2data1"], 'r') as file:
            x, y = [], []
            for i in file:
                i = list(map(float, i.split(sep=',')))
                x.append(i[:2])
                y.append(i[-1])
            x, y = np.array(x), np.array(y)
        return x, y

    def load_cardio(self):
        df = pd.read_csv(self.paths["Cardio"], sep=';')
        return df

    def load_hpart(self):
        train, test = pd.read_csv(self.paths["hpart_train"]), pd.read_csv(self.paths["hpart_test"])
        return train, test

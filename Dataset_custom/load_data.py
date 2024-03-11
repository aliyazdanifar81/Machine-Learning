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
            "Cardio": "/home/ali/Desktop/python/Machine Learning/Dataset_custom/Cardio/cardio_train.csv"
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
        x_train, y_train = self.__titanic_train()
        x_test, simple_test = self.__titanic_test()
        return x_train, y_train, x_test, simple_test

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
        x = df.drop(["id", "cardio"], axis=1)
        y = df["cardio"]
        return x, y

    # Private Functions
    def __titanic_train(self):
        train = pd.read_csv(self.paths["Titanic_train"])
        x_train, y_train = train.drop(["PassengerId", "Name"], axis=1), train["Survived"]
        mode = x_train.mode().iloc[0]
        mode["Cabin"] = "B96"
        x_train = x_train.fillna(mode)
        encoded = LabelEncoder()
        x_train["Cabin"] = encoded.fit_transform(x_train["Cabin"])
        x_train["Ticket"] = encoded.fit_transform(x_train["Ticket"])
        x_train["Sex"] = encoded.fit_transform(x_train["Sex"])
        x_train["Embarked"] = encoded.fit_transform(x_train["Embarked"])
        return x_train, y_train

    def __titanic_test(self):
        test = pd.read_csv(self.paths["Titanic_test"])
        simple = test
        test = test.drop('Name', axis=1)
        mode = test.mode().iloc[0]
        mode["Cabin"] = "B57"
        test = test.fillna(mode)
        encoded = LabelEncoder()
        test["Cabin"] = encoded.fit_transform(test["Cabin"])
        test["Ticket"] = encoded.fit_transform(test["Ticket"])
        test["Sex"] = encoded.fit_transform(test["Sex"])
        test["Embarked"] = encoded.fit_transform(test["Embarked"])
        return test, simple

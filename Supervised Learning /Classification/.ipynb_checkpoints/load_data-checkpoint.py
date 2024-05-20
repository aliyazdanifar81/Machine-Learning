import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class LoadModel:
    def __init__(self):
        self.paths = {
            "house": "./Dataset_custom/houses.txt",
            "Titanic_train": "./../Dataset_custom/Titanic/train.csv",
            "Titanic_test": "./../Dataset_custom/Titanic/test.csv"
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
        x_test, out_test = self.__titanic_test()
        return x_train, y_train, x_test, out_test

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
        out_test = test
        test = test.drop('Name', axis=1)
        mode = test.mode().iloc[0]
        mode["Cabin"] = "B57"
        test = test.fillna(mode)
        encoded = LabelEncoder()
        test["Cabin"] = encoded.fit_transform(test["Cabin"])
        test["Ticket"] = encoded.fit_transform(test["Ticket"])
        test["Sex"] = encoded.fit_transform(test["Sex"])
        test["Embarked"] = encoded.fit_transform(test["Embarked"])
        return test, out_test
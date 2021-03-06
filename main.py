import sklearn as sk
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

#x is the data given, y to be predicted, train, valid, and test are the fractions to be trained, validated, and tested on
def split(x, y, train = 0.8, valid = 0): 
    test = 1 - train - valid
    x_trainy, x_test, y_trainy, y_test = train_test_split(x, y, (train+valid), test)
    x_train, x_valid, y_train, y_valid = train_test_split(x_trainy, y_trainy, train/(train+valid),valid/(train+valid))

    return x_train, x_valid, x_test, y_train, y_valid, y_test

data = pd.read_excel("Digital_Data.xlsx")
data.drop(["Project Ref No."], axis = 1)
print(data.head())

if (__name__ == "__main__"):
    pass
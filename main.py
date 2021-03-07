import sklearn as sk
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from sklearn import linear_model

## For Data visualisation 
import matplotlib.pyplot as plt


#x is the data given, y to be predicted, train, valid, and test are the fractions to be trained, validated, and tested on
def split(x, y, train = 0.8, test = 0): 
    valid = 1 - train - test
    x_trainy, x_valid, y_trainy, y_valid = train_test_split(x, y, train_size = (train+test), test_size = valid)
    if test > 0:
        x_train, x_test, y_train, y_test = train_test_split(x_trainy, y_trainy, train_size = train/(train+test), test_size = test/(train+test))
    else:
        x_train, y_train =  x_trainy, y_trainy
        x_train, y_train =  x_trainy, y_trainy
        x_test, y_test = 0, 0
    return x_train, x_valid, x_test, y_train, y_valid, y_test

def score_RandomForest(x_train, x_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=100)
    model.fit(x_train, y_train)
    preds = model.predict(x_valid)
    return model, mean_absolute_error(y_valid, preds)

data = pd.read_excel("Digital_Data.xlsx")
data = data.drop(["Project Ref No."], axis = 1)
label_encoder = LabelEncoder()
s = (data.dtypes == 'object')
cat_cols = list(s[s].index)
for col in cat_cols:
    data[col] = label_encoder.fit_transform(data[col].astype(str))
# Dropped NaN values
data = data.dropna()
print(data.head(15))

## Keep getting different variables with the highest importance??
def LinReg (X, y, function):
    feature_list, importances, feature_importances = function
    sorted_by_second = sorted(feature_importances, key=lambda tup: tup[1], reverse=True)
    ## Edit ind_var to only use indepent variables 
    ind_var = ['Operator', 'Water depth', 'a1', 'Turbine rating (kW)', 'Blade length (m)', 'Tower height (m)', 'Built duration', 'Type', 'Metocean', 'Region']
    X = X[ind_var]
    regr = linear_model.LinearRegression()
    regr.fit(X,y)
    pass

def feature_importance(model, x_train):
    feature_list = list(x_train.columns)
    importances = model.feature_importances_
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
    return feature_list, importances, feature_importances

## Plotting the result of feature importance of RandomForest on graph
def plot(function):
    feature_list, importances, feature_importances = function
    plt.style.use('seaborn')
    x_values = list(range(len(importances)))
    plt.bar(x_values, importances, orientation = 'vertical')
    plt.xticks(x_values, feature_list, rotation=90)
    plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances')
    plt.tight_layout()
    plt.show()                         

if (__name__ == "__main__"):
    x = data.drop(columns = ['Nacelle Weights', 'Single Blade Weight (te)'])
    y = data[['Nacelle Weights', 'Single Blade Weight (te)']]
    x_train, x_valid, x_test, y_train, y_valid, y_test = split(x,y)
    model, mae = score_RandomForest(x_train, x_valid, y_train, y_valid)

## For Data visualisation 
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sn
from adspy_shared_utilities import plot_class_regions_for_classifier_subplot



data = pd.read_excel("Digital_Data.xlsx")
data.drop(["Project Ref No."], axis = 1)
print(data.head())

#x is the data given, y to be predicted, train, valid, and test are the fractions to be trained, validated, and tested on
def split(x, y, train = 0.8, valid = 0): 
    test = 1 - train - valid
    x_trainy, x_test, y_trainy, y_test = train_test_split(x, y, (train+valid), test)
    x_train, x_valid, y_train, y_valid = train_test_split(x_trainy, y_trainy, train/(train+valid),valid/(train+valid))

    return x_train, x_valid, x_test, y_train, y_valid, y_test

## Plotting the result of Random Classification on graph
def plot(classifier, x_train, x_valid, x_test, y_train, y_valid, y_test):
    plot_class_regions_for_classifier_subplot(classifier, x_train, y_train,None,
                                                None)                           


if (__name__ == "__main__"):
    pass
    
    ## Identify the features of importance from Random Forest
    feature_importance = feature_importance(model, x_train)
    plot(feature_importance)
    ## Multivariable Linear Regression
    LinReg(x, y, feature_importance)

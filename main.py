import sklearn as sk
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

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



## Plotting the result of Random Classification on graph
def plot(model, x_train, x_valid, x_test, y_train, y_valid, y_test):
    feature_list = list(x_train.columns)
    print(feature_list)
    importances = model.feature_importances_
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
    print (feature_importances)
    # Set the style
    plt.figure(111)
    plt.style.use('seaborn')# list of x locations for plotting
    x_values = list(range(len(importances)))# Make a bar chart
    plt.bar(x_values, importances, orientation = 'vertical')# Tick labels for x axis
    plt.xticks(x_values, feature_list, rotation=90)# Axis labels and title
    plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances')
    plt.tight_layout()
    plt.show()                         

if (__name__ == "__main__"):
    x = data.drop(columns = ['Nacelle Weights', 'Single Blade Weight (te)'])
    y = data[['Nacelle Weights', 'Single Blade Weight (te)']]
    x_train, x_valid, x_test, y_train, y_valid, y_test = split(x,y)
    model, mae = score_RandomForest(x_train, x_valid, y_train, y_valid)
    plot(model, x_train, x_valid, x_test, y_train, y_valid, y_test)
# Loads, splits and saves the data from kaggle files.

import pandas as pd
from sklearn.model_selection import train_test_split

# (X_train, y_train), (X_test, y_test) = mnist.load_data()
mnist = pd.read_csv('.')
x_train = mnist.iloc[:, 1:]
y_train = mnist.iloc[:, 0]
X_train, X_val, Y_train, Y_val = train_test_split(x_train, y_train, test_size=0.1)


X_train.to_csv('X_train_eval.csv', header=None, index=None)
Y_train.to_csv('Y_train_eval.csv', header=None, index=None)
X_val.to_csv('X_val_eval.csv', header=None, index=None)
Y_val.to_csv('Y_val_eval.csv', header=None,index=None)


# Baseline


from sklearn.linear_model import LogisticRegression
import pandas as pd
from load_data import *


def load_split():
    mnist = pd.read_csv('.')
    x_train = mnist.iloc[:, 1:]
    y_train = mnist.iloc[:, 0]
    return x_train, y_train


def log_reg_kaggle(x_train, y_train, x_test):
    log_reg = LogisticRegression(solver='lbfgs', multi_class='ovr')
    log_reg.fit(x_train, y_train)
    predict = log_reg.predict(x_test)
    with open('kaggle_logreg_predict.csv', 'w') as f:
        for idx, label in enumerate(predict):
            print(idx + 1, label, file=f)


if __name__ == '__main__':
    X_train, Y_train = load_split()
    plt_img = plot_learning_curve(estimator=LogisticRegression(), title='Learning Curve - Logistic Regression', X=X_train, y= Y_train, n_jobs=4)
    plt_img.savefig('log_reg.jpg')

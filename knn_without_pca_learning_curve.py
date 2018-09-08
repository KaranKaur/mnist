from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from load_data import *


def load_split():
    mnist = pd.read_csv('.')
    x_train = mnist.iloc[:, 1:]
    y_train = mnist.iloc[:, 0]
    return x_train, y_train


if __name__ == '__main__':
    X_train, Y_train = load_split()
    plt_img = plot_learning_curve(estimator=KNeighborsClassifier(n_neighbors=3), title='Learning Curve - kNN',
                                  X=X_train, y=Y_train, n_jobs=4)
    plt_img.savefig('kNN_lc.jpg')

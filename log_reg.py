# Baseline


from sklearn.linear_model import LogisticRegression
import pandas as pd


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
    test_data = pd.read_csv('.')
    X_test = test_data.iloc[:, :]
    log_reg_kaggle(X_train, Y_train, X_test)

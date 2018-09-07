import pandas as pd
from sklearn.neighbors import KNeighborsClassifier


def load_split():
    mnist = pd.read_csv('.')
    x_train = mnist.iloc[:, 1:]
    y_train = mnist.iloc[:, 0]
    print('data read!')
    return x_train, y_train


def knn_classifier(data_train, label_train, data_test, k):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(data_train, label_train)
    print('Model fitted!')
    predictions = model.predict(data_test)
    print('Predictions done!')
    with open('kaggle_kNN_predict.csv', 'w') as f:
        for idx, label in enumerate(predictions):
            print(idx + 1, label, file=f)


if __name__ == '__main__':
    X_train, Y_train = load_split()
    test_data = pd.read_csv('.')
    X_test = test_data.iloc[:, :]
    knn_classifier(X_train, Y_train, X_test, 3)

from load_data import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pca import *
from time import time
import pandas as pd


def load_kaggle_data():
    #training data
    mnist_train = pd.read_csv('.')
    data_train = mnist_train.iloc[:, 1:]
    label_train = mnist_train.iloc[:, 0]

    x_train, x_val, y_train, y_val = train_test_split(data_train, label_train, test_size=0.2)
    # x_test = pd.read_csv('.')
    return x_train, y_train, x_val, y_val


def load_kaggle_data_train():
    #training data
    mnist_train = pd.read_csv('.')
    x_train = mnist_train.iloc[:, 1:]
    y_train = mnist_train.iloc[:, 0]

    #test data
    x_test = pd.read_csv('.')
    return x_train, y_train, x_test


# 80% training (20% of training is validation data) and 20% testing data.
def knn_class_val(data_train, label_train, data_val, label_val):
    k_val = range(1, 11, 1)
    res = []
    for i in k_val:
        model = KNeighborsClassifier(n_neighbors=i)
        model.fit(data_train, label_train)
        score = model.score(data_val, label_val)
        print("With k = %d nearest neighbors, the accuracy = %.2f%%" % (i, score * 100))
        res.append(score)

    fig = plt.figure()
    plt.plot(k_val, res, 'r--')
    fig.suptitle('k vs Accuracy', fontsize=18)
    plt.xlabel('k', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    fig.savefig('kvsaccu.jpg')
    print('Saved the Image')

    # get the max accuracy
    max_acc_idx = res.index(max(res))
    max_acc = max(res)
    k_max_acc = k_val[max_acc_idx]
    print('K with max accuracy is %d and accuracy on validation set is %.2f' % (k_max_acc, max_acc))
    return k_max_acc


def knn_classifier(data_train, label_train, data_test, label_test, k):
    model = KNeighborsClassifier(n_neighbors=k)
    t1 = time()
    model.fit(data_train, label_train)
    # prediction = model.predict(data_test)
    score = model.score(data_test, label_test)
    t2 = time()
    time_diff = t2 - t1
    print('Time taken: %.2f' % time_diff)
    print("With k = 3 nearest neighbors, the accuracy = %.2f%%" % (score * 100))


def predict_knn_classifier(data_train, label_train, data_test, k):
    model = KNeighborsClassifier(n_neighbors=k)
    t1 = time()
    model.fit(data_train, label_train)
    prediction = model.predict(data_test)
    t2 = time()
    time_diff = t2 - t1
    print('Time taken: %.2f' % time_diff)
    return prediction


if __name__ == '__main__':
    X_train, Y_train, X_test = load_kaggle_data_train()
    scaled_x_train, scaled_x_test = scale_transform(X_train, X_test)
    transfd_x_train, transfd_x_test = pca_method(scaled_x_train, scaled_x_test)
    predictions = predict_knn_classifier(transfd_x_train, Y_train, transfd_x_test, 3)
    df = pd.DataFrame(predictions)
    df.to_csv("kaggle_predict_knn+pca.csv")
    print('Saved the predictions!')

    # X_train, Y_train, X_val, Y_val= load_kaggle_data()
    # val = [784, 0.99, 0.98, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 1]
    # for i in val:
    #     transfd_x_train, transfd_x_test = pca_method(i, scaled_x_train, scaled_x_test)
    #     knn_classifier(transfd_x_train, Y_train, transfd_x_test, Y_val, 3)

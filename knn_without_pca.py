import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from time import time


def load_split():
    mnist = pd.read_csv('.')
    x_data = mnist.iloc[:, 1:]
    y_data = mnist.iloc[:, 0]
    x_train, x_val, y_train, y_val = train_test_split(x_data.values, y_data.values, test_size=0.2)
    return x_train, x_val, y_train, y_val


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
    predictions = model.predict(data_test)
    t2 = time()
    time_diff = t2-t1
    print("Run time for kNN without PCA: %f" % time_diff)
    print("Performance on Test Data...")
    print(classification_report(label_test, predictions))
    with open('kaggle_kNN_predict.csv', 'w') as f:
        for idx, label in enumerate(predictions):
            print(idx + 1, label, file=f)


if __name__ == '__main__':
    X_train, X_val, Y_train, Y_val = load_split()
    #x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.2)
    #k_val_best = knn_class_val(X_train, Y_train, X_val, Y_val)
    knn_classifier(X_train, Y_train, X_val, Y_val, 3)
    #plt_img = plot_learning_curve(estimator=KNeighborsClassifier(n_neighbors=3), title='Learning Curve - k-NN',X=x_train, y=y_train, n_jobs=4)
    #plt_img.savefig('k-nn.jpg')

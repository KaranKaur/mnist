from load_data import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



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
    model.fit(data_train, label_train)
    prediction = model.predict(data_test)
    print("Performance on Test Data...")
    print(classification_report(label_test, prediction))


if __name__ == '__main__':
    X_train, X_test, Y_train, Y_test = load_split()
    x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.2)
    #k_val_best = knn_class_val(x_train, y_train, x_val, y_val)
    #knn_classifier(x_train, y_train, X_test, Y_test, k_val_best)
    plt_img = plot_learning_curve(estimator=KNeighborsClassifier(n_neighbors=3), title='Learning Curve - k-NN', X=x_train, y= y_train, n_jobs=4)
    plt_img.savefig('k-nn.jpg')
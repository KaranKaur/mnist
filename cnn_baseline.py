# from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
import pandas as pd
from sklearn.model_selection import train_test_split


# load the data files

def plot_figure():
    plt.subplot(221)
    plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))
    plt.subplot(222)
    plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))
    plt.subplot(223)
    plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))
    plt.subplot(224)
    plt.imshow(X_train[3], cmap=plt.get_cmap('gray'))
    plt.show()


np.random.seed(7)

def get_data():
    #read training data
    mnist_train = pd.read_csv('.')
    x_train = mnist_train.iloc[:, 1:]
    y_train = mnist_train.iloc[:, 0]

    #read test data
    mnist_test = pd.read_csv('.')
    x_test = mnist_test.iloc[:, :]

    return x_train.values, y_train.values, x_test.values


def baseline_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(1, 28, 28)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def main():
    X_train, Y_train, X_test = get_data()
    print(X_train.shape)

    # reshape to be [samples][pixels][width][height]
    x_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
    x_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

    # normalize
    x_train /= 255
    x_test /= 255
    y_train = np_utils.to_categorical(y_train)
    num_classes = y_train.shape[1]


    model = baseline_model()
    model.fit(x_train, y_train,batch_size=128, nb_epoch=10, verbose=1)
    #scores = model.evaluate(x_test, y_test, verbose=0)
    #print("CNN Error: %.2f%%" % (100 - scores[1] * 100))
    print('Fitted the model!')
    predict = model.predict_classes(x_test, batch_size=128, verbose=1)
    return predict

if __name__ == '__main__':
    predictions = main()
    df = pd.DataFrame(predictions)
    df.to_csv("predictions_base_cnn.csv")
    print('saved the predictions!')






import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D

from keras.callbacks import ReduceLROnPlateau
from keras.utils import np_utils
import pandas as pd



def plot_figure(X_train):
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


def plot_accuracy(history):
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='lower right')
    plt.savefig('Accuracy CNN Model.jpg')

def plot_loss(history):
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('Loss CNN Model.jpg')


def get_data():
    #training data
    mnist_train = pd.read_csv('.')
    x_train = mnist_train.iloc[:, 1:]
    y_train = mnist_train.iloc[:, 0]

    #test data
    mnist_test = pd.read_csv('.')
    x_test = mnist_test.iloc[:, :]

    return x_train.values, y_train.values, x_test.values


def baseline_model():
    model = Sequential()
    model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(1, 28, 28), padding='Same'))
    model.add(Conv2D(32, (5, 5), activation='relu', padding='Same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(1, 28, 28), padding='Same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='Same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def save_model(model):
    model_json = model.to_json()
    with open("cnn_model.json", "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model.save_weights("cnn_model.h5")
    print("Saved model to disk")

    # # load json and create model
    # json_file = open('model.json', 'r')
    # loaded_model_json = json_file.read()
    # json_file.close()
    # loaded_model = model_from_json(loaded_model_json)
    # # load weights into new model
    # loaded_model.load_weights("model.h5")
    # print("Loaded model from disk")


def main():
    X_train, y_train, X_test = get_data()
    #print(X_train.shape)

    # reshape to be [samples][pixels][width][height]
    x_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
    x_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

    # normalize
    x_train /= 255
    x_test /= 255
    y_train = np_utils.to_categorical(y_train)

    # learning rate reduction
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                                patience=3,
                                                verbose=1,
                                                factor=0.5,
                                                min_lr=0.00001)

    model = baseline_model()
    batch_size = 86

    # history = model.fit(x_train, y_train, batch_size=batch_size, epochs=30, verbose=2,
    #                     callbacks=[learning_rate_reduction])

    model.fit(x_train, y_train, batch_size=batch_size, epochs=30, verbose=2,
                         callbacks=[learning_rate_reduction])

    #print('Saving the plot of accuracy!')
    #plot_accuracy(history)
    #print('Saving the plot if Loss!')
    #plot_loss(history)

    #scores = model.evaluate(x_val, y_val, verbose=0)
    predictions = model.predict_classes(x_test, batch_size=128, verbose=1)
    #save_model(model)
    #print("CNN Error: %.2f%%" % (100 - scores[1] * 100))
    return predictions


if __name__ == '__main__':
    predictions = main()
    df = pd.DataFrame(predictions)
    df.to_csv("kaggle_predict_cnn.csv")
    print('Saved the predictions!')

import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten, BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.optimizers import RMSprop
from keras.callbacks import ReduceLROnPlateau
from keras.utils import np_utils
import pandas as pd
from keras import regularizers
import os
from keras.models import model_from_json


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


def plot1(history):
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.savefig('Accuracy-Model14.jpg')


def plot2(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.savefig('Loss-Model14.jpg')


def get_data():
    # ToDo : Change all paths to '.'
    # training data
    x_train = pd.read_csv('.')
    x_train = x_train.iloc[:, :]

    y_train = pd.read_csv('.')
    y_train = y_train.iloc[:, :]

    # validation data
    x_val = pd.read_csv('.')
    x_val = x_val.iloc[:, :]

    y_val = pd.read_csv('.')
    y_val = y_val.iloc[:, :]

    return x_train.values, y_train.values, x_val.values, y_val.values


def baseline_model():
    model = Sequential()
    model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(1, 28, 28), padding='Same'))
    model.add(Conv2D(32, (5, 5), activation='relu', padding='Same'))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(1, 28, 28), padding='Same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='Same'))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    # optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def save_model(model):
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model.save_weights("model.h5")
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
    x_train, y_train, x_val, y_val = get_data()

    # reshape to be [samples][pixels][width][height]
    x_train = x_train.reshape(x_train.shape[0], 1, 28, 28).astype('float32')
    x_val = x_val.reshape(x_val.shape[0], 1, 28, 28).astype('float32')

    # normalize
    x_train /= 255
    x_val /= 255
    y_train = np_utils.to_categorical(y_train)
    y_val = np_utils.to_categorical(y_val)
    # num_classes = y_train.shape[1]

    # learning rate reduction
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                                patience=3,
                                                verbose=1,
                                                factor=0.5,
                                                min_lr=0.00001)

    model = baseline_model()
    batch_size = 86
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=25, verbose=2,
                        validation_data=(x_val, y_val),
                        callbacks=[learning_rate_reduction])
    plot1(history)
    plot2(history)
    scores = model.evaluate(x_val, y_val, verbose=0)
    save_model(model)
    print("CNN Error: %.2f%%" % (100 - scores[1] * 100))


if __name__ == '__main__':
    main()

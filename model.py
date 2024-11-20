import keras
from keras.models import Sequential
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Activation, Dropout, Flatten, LSTM

def CnnModel(input_shape, labels):
    model = Sequential()
    model.add(Input(input_shape))

    model.add(Conv2D(32, (7, 7)))
    model.add(MaxPooling2D((3, 3), strides=(3, 3)))
    model.add(Activation('relu'))

    model.add(Conv2D(64, (7, 7), padding="same"))
    model.add(MaxPooling2D((3, 3), strides=(3, 3)))
    model.add(Activation('relu'))

    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dropout(rate=0.4))

    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(rate=0.4))

    model.add(Dense(labels))
    model.add(Activation('softmax'))

    return model

def LSTMModel(input_shape, labels):
    model = Sequential()
    model.add(Input(input_shape))

    model.add(LSTM(100))
    model.add(Dropout(rate=0.3))

    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(rate=0.3))

    model.add(Dense(labels))
    model.add(Activation('softmax'))

    return model
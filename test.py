
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

import pygame, sys, conv
from PIL import Image
from numpy import array

def make_model(weights_path):
    model = Sequential()

    model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(96, 96, 1)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Convolution2D(64, 2, 2, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Convolution2D(128, 2, 2, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dense(500, activation='sigmoid'))
    model.add(Dense(500, activation='sigmoid'))

    model.add(Dense(30, activation='linear'))
    model.load_weights(weights_path)
    return model

def load(f):

    answers = []
    images = []
    f.readline()

    for line in f:

        temp = line.split(',')
        if '' in temp:
            continue
        vals = []
        for val in temp[:-1]:
            vals.append(float(val))
        answers.append(vals)
        images.append(conv.string_to_np(temp[-1]))
    return array(answers), (images)

model = make_model('other.h5')

f = open('training.csv', 'r')
answers, images = load(f)

print(model.predict(array([images[0]])))

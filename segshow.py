
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

import pygame, sys, conv, imgbench as ibench
from PIL import Image
from numpy import array
import numpy as np
import pygame.camera
import time
import constants

def make_model(weights_path):
    model = Sequential()

    model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(constants.height, constants.width, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(64, 2, 2, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(128, 2, 2, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(256, 2, 2, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(1000, activation='relu'))

    model.add(Dense(1, activation='relu'))

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

def draw_rect(screen, x1, y1, x2, y2):
    points = []

    points.append((x1, y1))
    points.append((x2, y1))
    points.append((x2, y2))
    points.append((x1, y2))
    points.append((x1, y1))
    pygame.draw.lines(screen, pygame.Color(50, 180, 255, 255), True, points, 1)

def avg(pixel):
    return int((int(pixel[0]) + int(pixel[1]) + int(pixel[2]))/3)

def surf_to_np(surface):

    out = pygame.surfarray.array3d(surface).reshape((constants.height, constants.width, 3))

    return out

def grey_to_pyg(arr):
    out = np.zeros((96, 96, 3))

    for y, row in enumerate(out):
        for x, col in enumerate(row):
            out[y][x][0] = arr[y][x]
            out[y][x][1] = arr[y][x]
            out[y][x][2] = arr[y][x]
    return out


model = make_model('seg.h5')

# f = open('training.csv', 'r')
images, answers = conv.load_dataset(scale=.125, max_id=28, min_id=27)

nfaces = conv.load_nface()

ibench.to_PIL(images[0]).show()
ibench.to_PIL(nfaces[0]).show()

print(model.predict(array([images[0]])))
print(model.predict(array([nfaces[0]])))


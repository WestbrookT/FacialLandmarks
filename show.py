
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


    model.add(Dense(60, activation='relu'))
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


model = make_model('flip.h5')

# f = open('training.csv', 'r')
id = 29
images, answers = conv.load_dataset(scale=.125, max_id=id+1, min_id=id)
#images, answers = conv.load_my_set(19)
# images = images[-40:]
# answers = answers[-40:]
#f.readline()
#line = f.readline()


keys = ['FO-B', 'FO-L', 'FO-R', 'LE-C', 'LEA-B', 'LEB-I', 'LEB-O', 'LEL-B', 'LEL-I', 'LEL-O', 'LEL-T', 'LN-C',
             'LN-O', 'M-IB', 'M-IT', 'M-OB', 'M-OL', 'M-OR', 'M-OT', 'N-C', 'RE-C', 'REA-B', 'REB-I', 'REB-O', 'REL-B',
             'REL-I', 'REL-O', 'REL-T', 'RN-C', 'RN-O']

pygame.init()

size = width, height = (constants.width, constants.height)

black = 0,0,0

pygame.camera.init()


font = pygame.font.SysFont('arial', 6)


screen = pygame.display.set_mode(size)

#print(images)
i = 1
bench = ibench.ImageBench(screen)
ksel = -1
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT: sys.exit()

        if event.type == pygame.KEYUP:

            if event.key == pygame.K_1:
                ksel += 1
                if ksel == len(keys):
                    ksel = -1
            else:
                i += 1
                vals = model.predict(array([img]))[0]
                print(vals.tolist())
                print(answers[i].tolist(), '\n')

    time.sleep(.01)
    #print(np_arr)
    img = images[i]
    '0' * (3 - len(str(i))) + str(i)
    # pyimg = pygame.image.load('puts/0001/00011{}.JPG'.format('0' * (3 - len(str(i))) + str(i)))
    # pyimg = pygame.transform.scale(pyimg, (constants.width, constants.height))
    # screen.blit(pyimg, pyimg.get_rect())



    vals = model.predict(array([img]))[0]
    #vals = answers[i].flatten()
    bench.redraw(images[i], vals.tolist())

    kid = 0
    for pos, key in enumerate(keys):
        if pos == ksel or ksel == -1:
            text = font.render(key, True, (255, 255, 255))
            screen.blit(text, (vals[kid], vals[kid+1]))
        kid += 2

    # #print(answers[0])
    # vals = answers[i-1].flatten()
    # vals = vals




    # for j, x in enumerate(vals):
    #     #print(vals)
    #
    #     if j % 2 == 1:
    #         continue
    #     y = vals[j+1]
    #     x = int(x)
    #     y = int(y)
    #     draw_rect(screen, x-1, y-1, x+1, y+1)


    #draw_rect(screen, 100, 100, 300, 300)
    pygame.display.flip()


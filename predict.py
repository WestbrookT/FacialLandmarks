
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

import pygame, sys, conv
from PIL import Image
from numpy import array
import numpy as np
import pygame.camera
import time, constants, imgbench as ibench

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

    model.add(Dense(60, activation='linear'))
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

    temp = pygame.surfarray.array3d(surface).reshape((constants.height, constants.width, 3))
    out = np.zeros((constants.height, constants.width, 3))

    for y, row in enumerate(temp):
        for x, col in enumerate(row):
            out[x][y] = temp[y][x]
    return out

def grey_to_pyg(arr):
    out = np.zeros((96, 96, 3))

    for y, row in enumerate(out):
        for x, col in enumerate(row):
            out[y][x][0] = arr[y][x]
            out[y][x][1] = arr[y][x]
            out[y][x][2] = arr[y][x]
    return out


model = make_model('small.h5')

# f = open('training.csv', 'r')
# answers, images = load(f)
# f.readline()
# line = f.readline()




pygame.init()

size = width, height = 300, 300

black = 0,0,0

pygame.camera.init()
device = pygame.camera.list_cameras()[0]
cam = pygame.camera.Camera(device, (constants.width, constants.height))
cam.start()
img = cam.get_image()




screen = pygame.display.set_mode(size)
print(type(screen))
#exit()


while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT: sys.exit()

    time.sleep(.1)


    img = cam.get_image()
    #screen.blit(img, img.get_rect())

    pil = ibench.to_PIL(img)
    pil = pil.crop((25, 5, 135, 115))
    #print(pil.size)
    #pil = pil.resize((constants.width, constants.height), resample=Image.LANCZOS)
    img = ibench.to_surface(pil)
    np_arr = ibench.to_array(img)
    temp = pygame.Surface((constants.width, constants.height))
    #pygame.surfarray.blit_array(temp, np_arr)
    predictable = array([np_arr])
    #print(np_arr)
    screen.blit(img, (0,0))



    vals = model.predict(predictable)[0]
    #print(vals)



    for j, x in enumerate(vals):
        #print(vals)

        if j % 2 == 1:
            continue
        y = vals[j+1]
        x = int(x)
        y = int(y)
        draw_rect(screen, x-1, y-1, x+1, y+1)


    #draw_rect(screen, 100, 100, 300, 300)
    pygame.display.flip()



from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

import pygame, sys, conv
from PIL import Image
from numpy import array
import numpy as np
import pygame.camera
import time, constants, imgbench as ibench
import dlib









def normalize(arr):
    """
    Linear normalization
    http://en.wikipedia.org/wiki/Normalization_%28image_processing%29
    """
    arr = arr.astype('float')
    # Do not touch the alpha channel
    for i in range(3):
        minval = arr[...,i].min()
        maxval = arr[...,i].max()
        if minval != maxval:
            arr[...,i] -= minval
            arr[...,i] *= (255.0/(maxval-minval))
    return arr









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


model = make_model('small4.h5')

# f = open('training.csv', 'r')
# answers, images = load(f)
# f.readline()
# line = f.readline()




pygame.init()

size = width, height = 960, 540

black = 0,0,0

pygame.camera.init()
device = pygame.camera.list_cameras()[0]
cam = pygame.camera.Camera(device, (1920, 1080))
cam.start()
img = cam.get_image()

detector = dlib.get_frontal_face_detector()


screen = pygame.display.set_mode(size)
print(type(screen))
#exit()

bench = ibench.ImageBench(screen)

i = 0
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT: sys.exit()

    #time.sleep(1.1)


    img = cam.get_image()
    picture = ibench.resize(img, 960, 540)
    img = ibench.to_array(picture)
    dets = detector(img, 0)
    #print(dets)
    #screen.blit(img, img.get_rect())


    #print(pil.size)
    #pil = pil.resize((constants.width, constants.height), resample=Image.LANCZOS)


    #print(vals)

    screen.blit(picture, (0,0))

    for j, d in enumerate(dets):
        #print(vals)

        center = d.center()
        width, height = d.width(), d.height()
        #print(center, width, height)

        pil = ibench.to_PIL(img)

        scale = .65

        left = center.x - width*scale
        top = center.y - height*scale
        right = center.x + width*scale
        bottom = center.y + height*scale

        pil = pil.crop((left, top, right, bottom))
        pil_temp = pil


        scale = constants.width / pil.size[0]

        predictable_image = pil.resize((constants.width, constants.height), resample=Image.LANCZOS)


        # arr = normalize(ibench.to_array(pil))
        # predictable = array([arr])
        predictable = array([ibench.to_array(predictable_image)])




        predictions = model.predict(predictable)[0]*(1/scale)

        bounding_box = [(left, top), (right, top), (right, bottom), (left, bottom), (left, top)]
        points = predictions.tolist()
        points.append(bounding_box)
        #print(predictions.tolist())


        bench.redraw(pil, points, top_left=(left, top))
        #print(predictions.tolist())





    #print('\n\n')

    #draw_rect(screen, 100, 100, 300, 300)
    pygame.display.flip()
    screen.fill(black)


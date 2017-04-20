import yaml





import pygame, sys, conv
from PIL import Image
from numpy import array
import numpy as np
import pygame.camera
import time



def draw_rect(screen, x1, y1, x2, y2):
    points = []

    points.append((x1, y1))
    points.append((x2, y1))
    points.append((x2, y2))
    points.append((x1, y2))
    points.append((x1, y1))
    pygame.draw.lines(screen, pygame.Color(50, 180, 255, 255), True, points, 2)

def avg(pixel):
    return int((int(pixel[0]) + int(pixel[1]) + int(pixel[2]))/3)

def surf_to_np(surface):

    temp = pygame.surfarray.array3d(surface).reshape((96, 96, 3))
    out = np.zeros((96, 96, 1))

    for y, row in enumerate(temp):
        for x, col in enumerate(row):
            out[y][x] = avg(col)
    return out

def grey_to_pyg(arr):
    out = np.zeros((96, 96, 3))

    for y, row in enumerate(out):
        for x, col in enumerate(row):
            out[y][x][0] = arr[y][x]
            out[y][x][1] = arr[y][x]
            out[y][x][2] = arr[y][x]
    return out





f = open('00011001.yml')
f.__next__()
out = yaml.safe_load(f)
vals = []
for key in out:

    vals.append((out[key]['x'], out[key]['y']))


pygame.init()

size = width, height = 2000, 2000

black = 0,0,0

pygame.camera.init()





screen = pygame.display.set_mode(size)

i = 0
img = Image.open('00011001.JPG')



while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT: sys.exit()

        if event.type == pygame.KEYUP:
            i += 1

    time.sleep(.01)
    #print(np_arr)


    pyimg = pygame.image.load('00011001.JPG')
    screen.blit(pyimg, pyimg.get_rect())








    for j, val in enumerate(vals):
        #print(vals)

        x, y = val
        draw_rect(screen, x-1, y-1, x+1, y+1)


    #draw_rect(screen, 100, 100, 300, 300)
    pygame.display.flip()


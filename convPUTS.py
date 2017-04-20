from numpy import array
from PIL import Image
import pygame

def string_to_np(string):
    '''

    :param string:
    :return: An array for use in a keras model
    '''
    temp = []
    for i, val in enumerate(string.split()):
        temp.append(array([int(val)]))

    out = array(temp)
    out.resize((96, 96, 1))
    return out


def string_to_np_pil(string):
    '''

    :param string:
    :return: An array for use in pillow output
    '''
    temp = []
    for i, val in enumerate(string.split()):
        temp.append(array([int(val)]))

    out = array(temp)
    out.resize((96, 96))
    return out

def np_to_pil(numparray):

    return Image.fromarray(numparray.astype('uint8')).convert('RGB')


def pil_to_pyg(img):

    return pygame.image.fromstring(img.tobytes(), img.size, img.mode)

from numpy import array
import numpy as np
from PIL import Image
import pygame, os

import yaml, imgbench
import constants, random

def load_dataset(max_id=5, scale=1, min_id=1):
    ins = []
    outs = []
    for i in range(min_id, max_id):
        new_in, new_out = load_faces(i, scale=1)
        ins += new_in
        outs += new_out

    ins, outs = imgbench.augment_images(ins, outs, size=constants.width, crop_size=5, flatten=True, scale=scale)
   # print(ins[0].shape)
    outs = array(outs)
    return ins, outs

def load_faces(person_id, scale=1):
    print('ID:', person_id)
    """

    :param person_id: The id of the person, based on PUTS numbering
    :return: A 2-tuple of the face data and the landmark data for all photos

    The landmark data will have the keys sorted, and then the data returned in that sorted order. x, y, x1, y1, ...
    """

    person_id = str(person_id)
    face_folder = '0'*(4-len(person_id)) + person_id
    landmark_folder = 'L' + '0'*(3-len(person_id)) + person_id

    paths = os.listdir('puts/'+face_folder)

    images = []
    landmarks = []

    for path in paths:



        img_path = 'puts/{}/{}'.format(face_folder, path)
        landmark_path = 'puts/{}/{}'.format(landmark_folder, path[:-4]+'.yml')
        try:
            landmarks.append(landmark_file_to_list(landmark_path,scale=1))
            images.append(open_img_as_np(img_path, scale=1))
        except:
            pass



    return (images, landmarks)


def landmark_file_to_list(path, scale=1):
    f = open(path)
    f.__next__()
    data = yaml.safe_load(f)
    f.close()

    keys = sorted(data.keys())
    out = []

    for key in keys:
        out.append((data[key]['x']*scale, data[key]['y']*scale))
    #print(keys)

    return out

def string_to_np(string):
    '''

    :param string:
    :return: An array for use in a keras model
    '''
    temp = []
    for i, val in enumerate(string.split()):
        temp.append(array([int(val)]))

    out = array(temp)
    out.resize((2048, 1536, 3))
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
    out.resize((2048, 1536, 3))
    return out

def np_to_pil(numparray):

    return Image.fromarray(numparray.astype('uint8')).convert('RGB')


def pil_to_pyg(img):

    return pygame.image.fromstring(img.tobytes(), img.size, img.mode)


def open_img_as_np(image_path, scale=1):

    img = Image.open(image_path)
    img = img.resize((int(img.width*scale), int(img.height*scale)), resample=Image.LANCZOS)


    return np.asarray(img, dtype=np.uint8)

def load_nface():
    paths = os.listdir('nface')

    paths = ['nface/' + path for path in paths]
    print(paths)
    points = [[(0, 0)] for i in range(len(paths))]
    print(points)

    return imgbench.augment_images(paths, points, 10, size=110, scale=2)[0]


def load_my_set(count):

    images = []
    answers = []

    for i in range(0, count):
        resized = imgbench.resize('myface/{}.jpg'.format(i), constants.width, constants.height)
        images.append(resized)

        out = []

        with open('myl/{}.yml'.format(i), 'r') as f:
            data = yaml.safe_load(f)
            keys = sorted(data.keys())


            for key in keys:
                out.append((data[key]['x'], data[key]['y']))


        answers.append(out)

    return imgbench.augment_images(images, answers, 5, size=constants.width, flatten=True, scale=1.1)



# print(open_pil_as_np('00011001.JPG').shape)
# np_to_pil(open_pil_as_np('00011001.JPG')).show()


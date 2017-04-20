import conv
from numpy import array

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import rmsprop, sgd

import constants

rms = rmsprop(lr=.0001)
opt = sgd(lr=.03, momentum=.9, nesterov=True)






#print(images)

# training_answers, training_images = answers[:2100], images[:2100]
# test_answers, test_images = answers[1800:], images[1800:]

model = Sequential()

model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(constants.height, constants.width, 3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(64, 2, 2, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(128, 2, 2, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(256, 2, 2, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Convolution2D(512, 2, 2, activation='relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Convolution2D(1024, 2, 2, activation='relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Convolution2D(2048, 2, 2, activation='relu'))
# model.add(MaxPooling2D(pool_size=(4,4)))
# model.add(Convolution2D(2048, 2, 2, activation='relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Convolution2D(2048, 2, 2, activation='relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Convolution2D(2048, 2, 2, activation='relu'))





model.add(Flatten())
model.add(Dense(1000, activation='tanh'))
model.add(Dense(1000, activation='tanh'))

model.add(Dense(1, activation='tanh'))




model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
print(model.summary())

import numpy as np

for i in range(1, 2, 3):
    faces, answers = conv.load_dataset(scale=.125, max_id=i+1, min_id=i)
    answers = [[1] for i in range(len(faces))]
    nfaces = conv.load_nface()
    print(len(nfaces), len(faces))
    nanswers = [[-1] for i in range(len(nfaces))]

    images = np.append(faces, nfaces, axis=0)
    answers = np.append(answers, nanswers, axis=0)

    seed = 1
    np.random.seed(seed)
    np.random.shuffle(images)
    np.random.seed(seed)
    np.random.shuffle(answers)


    print('\n' * 5, 'Epoch:', i, '\n' * 5)
    model.fit(images, answers, nb_epoch=300, batch_size=16)

    model.save('seg.h5')
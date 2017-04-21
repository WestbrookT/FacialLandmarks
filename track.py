import conv
from numpy import array

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import rmsprop, sgd

import constants, random, imgbench as ibench, numpy as np

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
model.add(Dense(1000, activation='relu'))

model.add(Dense(1000, activation='relu'))


model.add(Dense(60, activation='relu'))




model.compile(optimizer=rms, loss='mse', metrics=['accuracy'])
model.load_weights('rescale.h5')
print(model.summary())




for i in range(1, 10):
    num = random.randint(1, 24)
    print('\n' * 5, 'Epoch:', i, '\n' * 5)

    images, answers = conv.load_dataset(scale=.125, max_id=num+3, min_id=num)
    images1, answers1 = conv.load_my_set(19)
    images = np.append(images, images1, axis=0)
    answers = np.append(answers, answers1, axis=0)



    ibench.shuffle_examples(images, answers)

    print('\n' * 5, 'Epoch:', i, '\n' * 5)
    model.fit(images, answers, nb_epoch=35, batch_size=64)

    model.save('rescale1.h5')

# images, answers = conv.load_my_set(19)
# ibench.shuffle_examples(images, answers)
# model.fit(images, answers, nb_epoch=30, batch_size=16)
# model.save('rescale.h5')

import conv
from numpy import array

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import rmsprop, sgd

rms = rmsprop(lr=.0001)
opt = sgd(lr=.03, momentum=.9, nesterov=True)
f = open('training.csv', 'r')

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


    return array(answers), array(images)

answers, images = load(f)
training_answers, training_images = answers[:2100], images[:2100]
test_answers, test_images = answers[1800:], images[1800:]

model = Sequential()

model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(2048, 1536, 3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(64, 2, 2, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(128, 2, 2, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(1000, activation='relu'))

model.add(Dense(30, activation='linear'))




model.compile(optimizer=rms, loss='mse', metrics=['accuracy'])
print(model.summary())


for i in range(0, 50):
    model.fit(images, answers, nb_epoch=100, batch_size=64)

    model.save('rms1.h5')
from flask import Flask, render_template, request, send_file
import imgbench as ibench, constants, base64
import pygame, numpy as np
from io import BytesIO, StringIO
from PIL import Image
import dlib
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Dense, Flatten

app = Flask(__name__)

bench = ibench.ImageBench(pygame.Surface((1, 1)))
detector = dlib.get_frontal_face_detector()



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
    print(model.summary())
    return model


model = make_model('small4.h5')




def serve_pil_image(pil_img):
    img_io = BytesIO()
    pil_img.save(img_io, 'JPEG', quality=70)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpeg')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/gp', methods=['POST'])
def get_points():

    b64 = request.form['b64']

    b64 = b64[len('data:image/jpeg;base64,'):]

    pil_img = Image.open(BytesIO(base64.b64decode(b64)))




    screen = pygame.Surface(pil_img.size)
    bench.update_surface(screen)
    bench.redraw(pil_img, top_left=(0, 0), new_points=[])

    img_arr = ibench.to_surface(pil_img)
    img_arr = ibench.to_array(img_arr)


    dets = detector(img_arr, 1)


    for j, d in enumerate(dets):
        #print(vals)

        center = d.center()
        width, height = d.width(), d.height()
        #print(center, width, height)

        pil = pil_img

        scale = .65

        left = center.x - width*scale
        top = center.y - height*scale
        right = center.x + width*scale
        bottom = center.y + height*scale

        pil = pil.crop((left, top, right, bottom))



        scale = constants.width / pil.size[0]

        predictable_image = pil.resize((constants.width, constants.height), resample=Image.LANCZOS)


        # arr = normalize(ibench.to_array(pil))
        # predictable = array([arr])
        predictable = np.array([ibench.to_array(predictable_image)])




        predictions = model.predict(predictable)[0]*(1/scale)

        bounding_box = [(left, top), (right, top), (right, bottom), (left, bottom), (left, top)]
        points = predictions.tolist()
        points.append(bounding_box)
        #print(predictions.tolist())


        bench.redraw(pil, points, top_left=(left, top))


    buf = BytesIO()
    out_img = ibench.to_PIL(screen)
    out_img.save(buf, format='JPEG')
    img_str = base64.b64encode(buf.getvalue())

    return 'data:image/jpeg;base64,' + str(img_str)[2:-1]
#'data:image/jpeg;base64,' +

@app.route('/t')
def t():
    return render_template('test.html')

@app.route('/test')
def test():
    return 'string mother fucker'

if __name__ == '__main__':
    app.run()
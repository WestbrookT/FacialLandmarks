from PIL import Image
import conv

f = open('training.csv', 'r')
f.readline()
i = 0

for line in f:
    line = line.split(',')
    data = line[-1]
    img = conv.np_to_pil(conv.string_to_np_pil(data))
    img.save('faces/{}.jpg'.format(i))
    i += 1




"""
The goal is to create an easy to use image system, to open images, and make certain they're in the correct
shape for use with numpy and or keras. There should be a system to display images easily, and to update the screen.
"""

import pygame, numpy as np, numbers, random
from PIL import Image

class ImageBench:
    """
    A system to easily and correctly draw points and images on the screen
    :param current_image: A numpy array of the current image
    :param points: A list containing tuples or lists of tuples, each tuple in (x, y) format
    :param surface: The location that things will be drawn to.
    :param camera: Possible camera to use as the image source
    :param color: pygame color
    """
    current_image = None

    points = []
    surface = None
    camera = None
    color = pygame.Color(100, 170, 250, 170)
    top_left = (0,0)

    def __init__(self, surface, current_image=None, points=None, center=None):
        """
        Creates an ImageBench object
        :param surface: The location to be drawn to
        :param current_image: An image in the form of a numpy array, a PIL Image, or a pygame surface, or a path string
        :param points: A list of points or point lists
        """
        self.surface = surface
        self.redraw(current_image, points)


    def redraw(self, new_source=None, new_points=None, new_surface=None, top_left=None):
        """

        :param new_source: An image in the form of a numpy array, a PIL Image, or a pygame surface, or a path string
        :param new_points: This is a list of (x, y) tuples, or if instead of a tuple used at any point in the list
                           there is another list lines will be drawn in order between those (x, y) tuples
        :param new_surface: The target to be drawn to, pygame surface
        :return: The screen that is drawn to
        """

        '''
        If there is no new image, the screen should just be rendered with the old image
        If there are no points then the old points will be used
        If surf is passed in then all things should be done to that surface rather than the surface contained within
        '''


        self.update_image(new_source)
        self.update_points(new_points)
        self.update_surface(new_surface)
        self.update_TL(top_left)

        self.draw_image()
        self.draw_points()


    def draw_lines(self, points):

        pygame.draw.lines(self.surface, self.color, False, points, 1)


    def draw_point(self, xy_tuple):
        """
        Draws a point on the internal surface
        :param xy_tuple: Length 2 tuple of ints
        :return: None
        """



        x, y = xy_tuple
        x = x + self.top_left[0]
        y = y + self.top_left[1]
        # x = self.image_center[0] - width // 2 + x
        # y = self.image_center[1] - height // 2 + x

        points = []
        size = 1
        points.append((x-size, y-size))
        points.append((x+size, y-size))
        points.append((x+size, y+size))
        points.append((x-size, y+size))
        points.append((x-size, y-size))
        pygame.draw.lines(self.surface, self.color, True, points, size)
        #print(points)

    def draw_points(self):
        """
        Takes no arguments, it just draws all of the internal points
        :return: None
        """
        if self.points is not None:
            cur_point = []
            flat = False
            for item in self.points:

                if flat and not isinstance(item, numbers.Number):
                    raise Exception('Points mismatch, got a single number for coordinates {}'.format(item))

                if isinstance(item, tuple) and len(item) == 2:
                    self.draw_point(item)
                elif isinstance(item, list):

                    self.draw_lines(item)
                elif isinstance(item, numbers.Number):
                    cur_point.append(item)
                    if len(cur_point) == 2:

                        self.draw_point(tuple(cur_point))
                        cur_point = []
                        flat = False
                    else:
                        flat = True
                else:
                    raise Exception('Invalid type for drawing points')

    def draw_image(self):
        """
        Takes no arguments, just draws the current internal image
        :return: None
        TODO: add the ability to specify the location of the blit
        """

        #self.surface.fill((0, 0, 0))
        if self.current_image is not None:
            image_surface = to_surface(self.current_image) #Converts the internal array to a drawable form (pygame surface)



            # x = self.image_center[0] - width//2
            # y = self.image_center[1] - height//2
            self.surface.blit(image_surface, self.top_left)


    def update_image(self, img_source):
        """
        Updates the internal image
        :param img_source: An image in the form of a numpy array, a PIL Image, or a pygame surface, or a path string
                            if the value is None or any falsey value nothing is done
        :return: None
        """
        if img_source is not None:
            img_array = to_array(img_source)
            self.current_image = img_array

    def update_points(self, new_points):
        """
        Updates the current list of points to be drawn
        :param new_points: List of point tuples or lists of point tuples
                           if the value is None or any falsey value nothing is done
        :return: None
        """
        if new_points is not None:
            self.points = new_points

    def update_surface(self, new_surface):
        """
        Updates the internala reference of the target surface
        :param new_surface: Pygame surface to be drawn to
                           if the value is None or any falsey value nothing is done
        :return: None
        """
        if new_surface:
            self.surface = new_surface

    def update_color(self, r, g, b, a):
        self.color = pygame.Color(r, g, b, a)

    def update_TL(self, x=None, y=None):

        if isinstance(x, tuple):
            self.top_left = x
        elif isinstance(x, int) and isinstance(y, int):

            self.top_left = (x, y)
        elif x is None and y is None:
            pass
        else:
            raise Exception("First parameter must be an (x, y) tuple, or both x, and y must be ints, got {}".format(type(x)))


def to_array(origin):
    """

    :param origin: Origin should be a PIL image, a path string, or a pygame surface
    :return: A numpy array
    """
    if isinstance(origin, Image.Image):
        """
        Check if the origin is a PIL image
        """
        #print(origin.size)
        #origin = origin.rotate(90, expand=True)
        origin = np.asarray(origin, dtype=np.uint8)
        #print(origin.shape)
        return origin
    elif isinstance(origin, str):
        return to_array(Image.open(origin))

    elif isinstance(origin, type(pygame.Surface((1,1)))):
        """
        Pygame surfaces are worked with in the same way as PIL images
        They have their axes swapped
        So on output to pygame surfaces they swapped again
        """

        return np.swapaxes(pygame.surfarray.array3d(origin), 0, 1)
    elif isinstance(origin, np.ndarray):
        """
        Return the array if it is already an array
        """
        return origin
    else:
        raise Exception('Incompatible Type of Object')

def to_PIL(origin):
    if not isinstance(origin, np.ndarray):
        origin = to_array(origin)
    return Image.fromarray(origin)

def to_surface(origin):
    if not isinstance(origin, np.ndarray):
        origin = to_array(origin)
    return pygame.surfarray.make_surface(np.swapaxes(origin, 0, 1))

def rgb_to_grey(array):
    raise Exception("Not currently implemented")

def grey_to_rgb(array):
    raise Exception("Not currently implemented")


def flip(origin, points):
    """

    :param origin: Can be an image in array, pil, or surface format
    :param points: List of (x, y) tuples
    :return: A new array of the image, and the new point set
    """

    if not isinstance(origin, Image.Image):
        origin = to_PIL(origin)

    origin = origin.transpose(Image.FLIP_LEFT_RIGHT)
    width, height = origin.size

    out = []
    for point in points:
        #print('width', width,'x', point[0], 'y', point[1])

        out.append((width-point[0]-1, point[1]))

    return to_array(origin), out




def crop_aug(pilimg, points, crop_size=50, center=None, size=1000, flatten=False, scale=1, intensity_fuzz=.1):
    """

    :param pilimg: A pillow image object
    :param points: A list of (x,y) tuples
    :param crop_size: Amount of pixels to crop off
    :param center: The location of the area to crop around
    :param size: The height and width of the end image
    :return: a tuple of two lists, one with the cropped images, another with its corresponding points
    """

    if not isinstance(pilimg, Image.Image):
        pilimg = to_PIL(pilimg)

    new_width, new_height = int(pilimg.width*scale), int(pilimg.height*scale)

    pilimg = pilimg.resize((new_width, new_height), resample=Image.LANCZOS)

    x = 0
    y = 0
    if center is None:
        x = pilimg.width//2
        y = pilimg.height//2
    else:
        x, y = center

    outimgs = []
    outpoints = []




    for xmod in [0, -1, 1]:

        for ymod in [0, -1, 1]:


            left = x - size//2 + xmod*crop_size
            right = x + size//2 + xmod*crop_size
            top = y - size//2 + ymod*crop_size
            bottom = y + size//2 + ymod*crop_size

            cropped = pilimg.crop((left, top, right, bottom))

            outimgs.append(to_array(cropped))

            curpts = []

            for point in points:

                xp, yp = point
                xp, yp = int(xp*scale), int(yp*scale)

                inside = (xp >= left and xp < right) and (yp >= top and yp < bottom)

                xp = (xp - left) if inside else 0
                yp = (yp - top) if inside else 0


                if flatten:
                    curpts.append(xp)
                    curpts.append(yp)
                else:
                    curpts.append((xp, yp))
            outpoints.append(curpts)

    # for i, image in enumerate(outimgs):
    #     break
    #     outimgs.append(darken(image, intensity_fuzz))
    #     outpoints.append(outpoints[i])
    #     outimgs.append(lighten(image, intensity_fuzz))
    #     outpoints.append(outpoints[i])

    return outimgs, outpoints

def augment_images(img_list, point_list, crop_size=50, center=None, size=1000, flatten=True, scale=1, scale_fuzz=.01, intensity_fuzz=.1):
    """
    Uses the crop_aug function to create a keras ready dataset
    :param img_list: Can be a list of images in array, pil, or surface format
    :param point_list: List of lists of point tuples (x, y) corresponding to each image
    :return: A tuple (images, points) where images is a numpy array of numpy arrays of images,
                and points is a numpy array of numpy arrays of points
    """
    out_images = []
    out_points = []

    for i, image in enumerate(img_list):

        new_images, new_points = crop_aug(image, point_list[i], crop_size, center, size, flatten, scale)

        out_images += new_images
        out_points += new_points

        new_images, new_points = crop_aug(image, point_list[i], crop_size, center, size, flatten, scale+scale_fuzz)

        out_images += new_images
        out_points += new_points

        new_images, new_points = crop_aug(image, point_list[i], crop_size, center, size, flatten, scale+scale_fuzz)

        out_images += new_images
        out_points += new_points

        # image, pts = flip(image, point_list[i])
        #print(point_list[i], pts)

        # new_images, new_points = crop_aug(image, pts, crop_size, center, size, flatten, scale)
        #
        # out_images += new_images
        # out_points += new_points



    if flatten:
        out_points = np.array(out_points)

    return np.array(out_images), out_points

def darken(image, fuzz):

    image = to_PIL(image)

    pixels = image.getdata()

    new_image = Image.new('RGB', image.size)
    new_image_list = []

    mult = 1.0 - fuzz

    for pixel in pixels:
        new_pixel = [int(pixel[0] * mult), int(pixel[1] * mult), int(pixel[2] * mult)]

        for i, val in enumerate(new_pixel):
            if val > 255:
                new_pixel[i] = 255
            elif val < 0:
                new_pixel[i] = 0

        new_image_list.append(tuple(new_pixel))
    new_image.putdata(new_image_list)
    return to_array(new_image)

def lighten(image, fuzz):
    return darken(image, -fuzz)


def resize(origin, width, height):
    origin_original = origin
    if not isinstance(origin, Image.Image):
        origin = to_PIL(origin)

    origin = origin.resize((width, height), resample=Image.LANCZOS)

    if isinstance(origin_original, Image.Image):
        return origin
    elif isinstance(origin_original, type(pygame.Surface((1,1)))):
        return to_surface(origin)
    elif isinstance(origin_original, np.ndarray):
        return to_array(origin)
    elif isinstance(origin_original, str):
        return to_array(origin)
    else:
        raise Exception('Got {} expected a surface, ndarray, or a pil image, or a str path'.format(type(origin_original)))

if __name__ == '__main__':
    # to_array(Image.open('test.jpg'))
    # to_array('test.jpg')
    # to_array(pygame.Surface((2,1)))
    #
    # img = Image.open('rgba.png')
    # img.show()
    img = pygame.image.load("face.JPG")

    # input('waiting...')
    # to_PIL(to_array('rgba.png')).show()
    import time

    pygame.init()
    size = w, h = 1500, 1000
    screen = pygame.display.set_mode(size)
    pts = [(300, 300), (600, 600), (900, 700)]

    print(flip(img, pts))

    cimgs, cpts = augment_images([img, img], [pts, pts], size=100, crop_size=5, scale=.125)

    bench = ImageBench(screen, img, pts)

    bench.redraw()
    pygame.display.flip()
    input('waiting...')

    img, pts = flip(img, pts)
    bench.redraw(img, pts)
    pygame.display.flip()
    input('waiting')


    print('yar', cpts[0])
    for i in range(len(cimgs)):
        bench.redraw(new_source=cimgs[i], new_points=cpts[i])
        pygame.display.flip()
        time.sleep(2)

def shuffle_examples(inputs, outputs, seed=1):

    np.random.seed(seed)
    print(inputs.shape)
    np.random.shuffle(inputs)
    print(inputs.shape)
    np.random.seed(seed)
    print(outputs.shape)
    np.random.shuffle(outputs)
    print(outputs.shape)
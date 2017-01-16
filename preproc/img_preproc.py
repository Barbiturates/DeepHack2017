import numpy as np
from skimage import color
from skimage import transform


def grayscale(image):
    """
    Grayscales the image
    :param image: 250x160x3 image
    :return: 250x160 image
    """

    return color.rgb2gray(image)


def downsample(image):
    """
    Downsamples the image
    :param image: 250x160 image
    :return: 110x84 image
    """
    return transform.resize(image, (110, 84))


def crop_image(image):
    """
    Crops the image
    :param image: 110x84 image
    :return: 84x84 image
    """

    return image[20:104, 0:84]


def preproc_img(image):
    """
    Turns the 250x160x3 image to
     84x84 gray image
    :param image: 250x160x3 image
    :return: 84x84 gray image
    """

    image = grayscale(image)
    image = downsample(image)
    image = crop_image(image)
    return image

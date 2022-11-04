import enum

import numpy as np
import tensorflow as tf


class ColorType(enum.Enum):
    GRAYSCALE = 1
    RGB = 3
    RGBA = 4


def build_images(count, color_type, image_width, image_height, *, to_tensor=False):
    n_channels = color_type.value
    if count == 1:
        size = (image_height, image_width, n_channels)
    else:
        size = (count, image_height, image_width, n_channels)
    image = np.random.randint(0, 256, size=size).astype(np.uint8)

    if to_tensor:
        image = tf.convert_to_tensor(image)

    return image


def build_image(color_type, image_width, image_height, *, to_tensor=False):
    return build_images(1, color_type, image_width, image_height, to_tensor=to_tensor)

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.utils import conv_utils
from keras.layers import Layer
from keras.engine import InputSpec

def bilinear_resize_images(x, size, method='bilinear'):
    new_size = tf.convert_to_tensor(size, dtype=tf.int32)
    resized = tf.image.resize_images(x, new_size)
    return resized

def resize_images(x, height_factor, width_factor, data_format):
    if data_format == 'channels_first':
        original_shape = K.int_shape(x)
        new_shape = tf.shape(x)[2:]
        new_shape *= tf.constant(np.array([height_factor, width_factor]).astype('int32'))
        x = permute_dimensions(x, [0, 2, 3, 1])
        x = tf.image.resize_bilinear(x, new_shape)
        x = permute_dimensions(x, [0, 3, 1, 2])
        x.set_shape((None, None, original_shape[2] * height_factor if original_shape[2] is not None else None,
                     original_shape[3] * width_factor if original_shape[3] is not None else None))
        return x
    elif data_format == 'channels_last':
        original_shape = K.int_shape(x)
        new_shape = tf.shape(x)[1:3]
        new_shape *= tf.constant(np.array([height_factor, width_factor]).astype('int32'))
        x = tf.image.resize_bilinear(x, new_shape)
        x.set_shape((None, original_shape[1] * height_factor if original_shape[1] is not None else None,
                     original_shape[2] * width_factor if original_shape[2] is not None else None, None))
        return x
    else:
        raise ValueError('Unknown data_format: ' + str(data_format))

class BilinearUpSampling2D(Layer):
    """ UpSampling2D uses nearest neighbor.
        This provides resize with bilinear interpolation.
    """

    def __init__(self, size=(2, 2), data_format=None, **kwargs):
        """ Constructor
        Parameters
        ----------
        target_shape: (width, height, channels) or (channels, width, height)
            Output shape instead of factors
        data_format: channels_last or channels_first
        """
        super(BilinearUpSampling2D, self).__init__(**kwargs)
        
        self.data_format = K.normalize_data_format(data_format)
        self.input_spec = InputSpec(ndim=4)
        self.size = conv_utils.normalize_tuple(size, 2, 'size')

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            height = self.size[0] * input_shape[2] if input_shape[2] is not None else None
            width = self.size[1] * input_shape[3] if input_shape[3] is not None else None
            return (input_shape[0],
                    input_shape[1],
                    height,
                    width)
        elif self.data_format == 'channels_last':
            height = self.size[0] * input_shape[1] if input_shape[1] is not None else None
            width = self.size[1] * input_shape[2] if input_shape[2] is not None else None
            return (input_shape[0],
                    height,
                    width,
                    input_shape[3])

    def call(self, inputs):
        return resize_images(inputs, self.size[0], self.size[1], self.data_format)

    def get_config(self):
        config = {'target_shape': self.target_shape, 'data_format': self.data_format}
        base_config = super(BilinearUpSampling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

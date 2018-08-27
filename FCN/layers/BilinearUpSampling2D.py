import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.engine import InputSpec

def bilinear_resize_images(x, size, method='bilinear'):
    new_size = tf.convert_to_tensor(size, dtype=tf.int32)
    resized = tf.image.resize_images(x, new_size)
    return resized

class BilinearUpSampling2D(Layer):
    """ UpSampling2D uses nearest neighbor.
        This provides resize with bilinear interpolation.
    """

    def __init__(self, target_shape=None, data_format=None, **kwargs):
        """ Constructor
        Parameters
        ----------
        target_shape: (width, height, channels) or (channels, width, height)
            Output shape instead of factors
        data_format: channels_last or channels_first
        """
        if data_format is None:
            data_format = K.image_data_format()
        assert data_format in ['channels_last', 'channels_first']
        self.data_format = data_format
        self.input_spec = InputSpec(ndim=4)
        self.target_shape = target_shape
        if self.data_format == 'channels_first':
            self.target_size = (target_shape[2], target_shape[3])
        elif self.data_format == 'channels_last':
            self.target_size = (target_shape[1], target_shape[2])
        super(BilinearUpSampling2D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_last':
            return (input_shape[0], self.target_size[0],
                    self.target_size[1], input_shape[3])
        else:
            return (input_shape[0], input_shape[1],
                    self.target_size[0], self.target_size[1])

    def call(self, inputs):
        return bilinear_resize_images(inputs, size=self.target_size, method='bilinear')

    def get_config(self):
        config = {'target_shape': self.target_shape, 'data_format': self.data_format}
        base_config = super(BilinearUpSampling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

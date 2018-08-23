from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, Concatenate, Activation, Add, BatchNormalization
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.regularizers import l2
from layers.BilinearUpSampling2D import BilinearUpSampling2D
import sys
import os

def FCN_VGG16(input_shape, train=False, weight_decay=0., classes=22):
    
    """ FCN Model based on VGG16
    Parameters
    ----------
    input_shape : (N, M), the shape for the InputLayer
    train       : use of train or not
    weight_decay: weight for L2 regularization
    Reterns
    --------
    FCN-VGG16 model
    """
    input_image = Input(shape=input_shape, name="input_1")

    # Block 1
    x = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), activation='relu', name="block1_conv1")(input_image)
    x = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), activation='relu', name="block1_conv2")(x)
    x = MaxPooling2D(strides=(2, 2), name="block1_pool")(x)
    # Block 2
    x = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), activation='relu', name="block2_conv1")(x)
    x = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), activation='relu', name="block2_conv2")(x)
    x = MaxPooling2D(strides=(2, 2), name="block2_pool")(x)
    # Block 3
    x = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), activation='relu', name="block3_conv1")(x)
    x = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), activation='relu', name="block3_conv2")(x)
    x = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), activation='relu', name="block3_conv3")(x)
    x = pool3 = MaxPooling2D(strides=(2, 2), name="block3_pool")(x)
    # Block 4
    x = Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), activation='relu', name="block4_conv1")(x)
    x = Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), activation='relu', name="block4_conv2")(x)
    x = Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), activation='relu', name="block4_conv3")(x)
    x = pool4 = MaxPooling2D(strides=(2, 2), name="block4_pool")(x)
    # Block 5
    x = Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), activation='relu', name="block5_conv1")(x)
    x = Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), activation='relu', name="block5_conv2")(x)
    x = Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), activation='relu', name="block5_conv3")(x)
    x = pool5 = MaxPooling2D(strides=(2, 2), name="block5_pool")(x)

    # fully-connected layer
    x = Conv2D(4096, (7, 7), kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), activation='relu', padding='same', name="fc1")(x)
    x = Dropout(0.5, name="fc1_dropout")(x)
    x = Conv2D(4096, (1, 1), kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), activation='relu', padding='same', name="fc2")(x)
    x = Dropout(0.5, name="fc2_dropout")(x)

    x = Conv2D(classes, (1, 1), kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), activation='relu', padding='same', name="fc_conv")(x)

    # p5
    p5 = Conv2D(classes, (1, 1), kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), activation='relu', padding='same', name= "p5_conv")(x)
    p5 = BilinearUpSampling2D(pool4.shape, data_format='channels_last', name="p5_upsampling")(p5)
    # p4
    p4 = Conv2D(classes, (1, 1), activation='relu', padding='same', name="p4_conv")(pool4)
    
    merge_p4p5 = Add(name="p4p5_add")([p5, p4])
    merge_p4p5 = BilinearUpSampling2D(pool3.shape, data_format='channels_last', name="p4p5_upsampling")(merge_p4p5)
    # p3
    p3 = Conv2D(classes, (3, 3), kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), padding='same', name="p3_conv")(pool3)
    
    merge_p3p4p5 = Add(name="p3p4p5_add")([merge_p4p5, p3])
    
    # upsampling
    output = BilinearUpSampling2D(input_image.shape, data_format='channels_last', name="output_upsampling")(merge_p3p4p5)
#        output = Conv2DTranspose(21, (3, 3), strides=(8, 8), activation='relu', padding='same', name="out_deconv")(merge_p3p4p5)
    output = Activation('softmax', name="out")(output)

    model = Model(input_image, output)

    vgg16 = VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
    
    if train:
        # reuse weight
        layer_names = ['block1_conv1', 'block1_conv2', 'block2_conv1', 'block2_conv2', 'block3_conv1', 'block3_conv2', 'block3_conv3',
            'block4_conv1', 'block4_conv2', 'block4_conv3', 'block5_conv1', 'block5_conv2', 'block5_conv3']
        
        for layer_name in layer_names:
            layer = model.get_layer(layer_name)
            layer.set_weights(vgg16.get_layer(layer_name).get_weights())
#                layer.trainable = False

    else:
        # load weights
        current_dir = os.path.dirname(os.path.realpath(__file__))
        learned_model_dir = os.path.join(current_dir, '../tmp/fcn_vgg16/')
        learned_model_dir = os.path.normpath(learned_model_dir)
        weights_path = os.path.join(
            learned_model_dir,
            #                'checkpoint_weights.hdf5'
            'model.hdf5'
        )
        model.load_weights(weights_path)

    return model

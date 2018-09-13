import sys
import os
import numpy as np
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, Concatenate, Activation, Add, BatchNormalization
from keras.applications.vgg16 import VGG16
from keras.regularizers import l2
from layers.BilinearUpSampling2D import BilinearUpSampling2D

def FCN_VGG16(input_shape, weight_decay=0., classes=21, target=16):
    
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

    # fully-convolution layer
    x = Conv2D(4096, (7, 7), kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), activation='relu', padding='same', name="fc1")(x)
    x = Dropout(0.2, name="fc1_dropout")(x)
    x = Conv2D(4096, (1, 1), kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), activation='relu', padding='same', name="fc2")(x)
    x = Dropout(0.2, name="fc2_dropout")(x)

    # p5
    p5 = Conv2D(classes, (1, 1), activation='linear', padding='valid', name= "p5_predict")(x)
    if target == 32:
        out_prediction = BilinearUpSampling2D((32, 32), data_format='channels_last', name="p5_upsampling")(p5)
    else:
        p5 = BilinearUpSampling2D((2, 2), data_format='channels_last', name="p5_upsampling")(p5)
        # p4
        p4 = Conv2D(classes, (1, 1), activation='linear', padding='valid', name="p4_predict")(pool4)
        
        merge_p4p5 = Add(name="p4p5_merge")([p5, p4])
        merge_p4p5 = Conv2D(classes, (1, 1), activation='linear', padding='valid', name="p4p5_predict")(merge_p4p5)
        if target == 16:
            out_prediction = BilinearUpSampling2D((16, 16), data_format='channels_last', name="p4p5_upsampling")(merge_p4p5)
        else:
            merge_p4p5 = BilinearUpSampling2D((2, 2), data_format='channels_last', name="p4p5_upsampling")(merge_p4p5)
            # p3
            p3 = Conv2D(classes, (1, 1), activation='linear', padding='valid', name="p3_predict")(pool3)
            
            merge_p3p4p5 = Add(name="p3p4p5_merge")([merge_p4p5, p3])
            
            # upsampling
            out_prediction = BilinearUpSampling2D((8, 8), data_format='channels_last', name="output_upsampling")(merge_p3p4p5)
    output = Activation('softmax', name="out")(out_prediction)

    model = Model(input_image, output)

    return model


def transferWeight(model, input_shape):
    vgg16 = VGG16(weights='imagenet', input_shape=input_shape)
    
    # reuse weight
    layer_names = ['block1_conv1', 'block1_conv2', 'block2_conv1', 'block2_conv2', 'block3_conv1', 'block3_conv2', 'block3_conv3',
                   'block4_conv1', 'block4_conv2', 'block4_conv3', 'block5_conv1', 'block5_conv2', 'block5_conv3', 'fc1', 'fc2']
                   
    trans = 0
    for layer in vgg16.layers:
        weights = layer.get_weights()
        if layer.name=='fc1':
            weights[0] = np.reshape(weights[0], (7,7,512,4096))
        elif layer.name=='fc2':
            weights[0] = np.reshape(weights[0], (1,1,4096,4096))
        if layer.name in layer_names:
            model.get_layer(layer.name).set_weights(weights)
            trans += 1

    print ("{} layers transfer complete!".format(trans))

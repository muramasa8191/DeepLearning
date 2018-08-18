from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, Concatenate, Activation, Add, BatchNormalization
from tensorflow.python.keras.applications.vgg16 import VGG16

import sys
import os

class FCN_VGG16():
    
    def build(self, input_shape, train=False):
        """
            
        """
        input_image = Input(shape=input_shape, name="input_1")

        # Block 1
        x = Conv2D(64, (3, 3), padding='same', kernel_initializer='VarianceScaling', activation='relu', name="block1_conv1")(input_image)
        x = Conv2D(64, (3, 3), padding='same', kernel_initializer='VarianceScaling', activation='relu', name="block1_conv2")(x)
        x = MaxPooling2D(strides=(2, 2), name="block1_pool")(x)
        # Block 2
        x = Conv2D(128, (3, 3), padding='same', kernel_initializer='VarianceScaling', activation='relu', name="block2_conv1")(x)
        x = Conv2D(128, (3, 3), padding='same', kernel_initializer='VarianceScaling', activation='relu', name="block2_conv2")(x)
        x = MaxPooling2D(strides=(2, 2), name="block2_pool")(x)
        # Block 3
        x = Conv2D(256, (3, 3), padding='same', kernel_initializer='VarianceScaling', activation='relu', name="block3_conv1")(x)
        x = Conv2D(256, (3, 3), padding='same', kernel_initializer='VarianceScaling', activation='relu', name="block3_conv2")(x)
        x = Conv2D(256, (3, 3), padding='same', kernel_initializer='VarianceScaling', activation='relu', name="block3_conv3")(x)
        x = pool3 = MaxPooling2D(strides=(2, 2), name="block3_pool")(x)
        # Block 4
        x = Conv2D(512, (3, 3), padding='same', kernel_initializer='VarianceScaling', activation='relu', name="block4_conv1")(x)
        x = Conv2D(512, (3, 3), padding='same', kernel_initializer='VarianceScaling', activation='relu', name="block4_conv2")(x)
        x = Conv2D(512, (3, 3), padding='same', kernel_initializer='VarianceScaling', activation='relu', name="block4_conv3")(x)
        x = pool4 = MaxPooling2D(strides=(2, 2), name="block4_pool")(x)
        # Block 5
        x = Conv2D(512, (3, 3), padding='same', kernel_initializer='VarianceScaling', activation='relu', name="block5_conv1")(x)
        x = Conv2D(512, (3, 3), padding='same', kernel_initializer='VarianceScaling', activation='relu', name="block5_conv2")(x)
        x = Conv2D(512, (3, 3), padding='same', kernel_initializer='VarianceScaling', activation='relu', name="block5_conv3")(x)
        x = pool5 = MaxPooling2D(strides=(2, 2), name="block5_pool")(x)

        # fully-connected layer
        #x = Conv2D(4096, (7, 7), activation='relu', padding='same', name='conv6')(x)
        #x = Dropout(0.5)(x)
        #x = Conv2D(4096, (1, 1), activation='relu', padding='same', name='conv7')(x)
        #x = Dropout(0.5)(x)

        x = Conv2D(21, (1, 1), activation='relu', padding='same')(x)

        # p5
        #        p5 = Conv2D(21, (1, 1), activation='relu', padding='same')(pool5)
        p5 = Conv2DTranspose(21, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
        # p4
        p4 = Conv2D(21, (1, 1), activation='relu', padding='same')(pool4)
        
        #        merge_p4p5 = Add()([p5, p4])
        merge_p4p5 = Add()([p5, p4])
        merge_p4p5 = BatchNormalization(axis=-1, momentum=0.99)(merge_p4p5)
        merge_p4p5 = Conv2DTranspose(21, (3, 3), strides=(2, 2), padding='same', activation='relu')(merge_p4p5)
        # p3
        p3 = Conv2D(21, (3, 3), padding='same', name="pool3_conv1")(pool3)
        
        #        merge_p3p4p5 = Add()([merge_p4p5, p3])
        merge_p3p4p5 = Add()([merge_p4p5, p3])
        merge_p3p4p5 = BatchNormalization(axis=-1, momentum=0.99)(merge_p3p4p5)
        
        # deconvolution
        output = Conv2DTranspose(21, (3, 3), strides=(8, 8), activation='relu', padding='same')(merge_p3p4p5)
        output = Activation('softmax')(output)

        model = Model(input_image, output)

        vgg16 = VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
        
        if train:
            # reuse weight
            layer_names = ['block1_conv1', 'block1_conv2', 'block2_conv1', 'block2_conv2', 'block3_conv1', 'block3_conv2', 'block3_conv3',
                'block4_conv1', 'block4_conv2', 'block4_conv3', 'block5_conv1', 'block5_conv2', 'block5_conv3']
            
            for layer_name in layer_names:
                layer = model.get_layer(layer_name)
                layer.set_weights(vgg16.get_layer(layer_name).get_weights())
                layer.trainable = False

        else:
            # load weights
            current_dir = os.path.dirname(os.path.realpath(__file__))
            learnt_model_dir = os.path.join(current_dir, '../tmp/fcn_vgg16/')
            learnt_model_dir = os.path.normpath(learnt_model_dir)
            weights_path = os.path.join(
                learnt_model_dir,
                #                'checkpoint_weights.hdf5'
                'model.hdf5'
            )
            model.load_weights(weights_path)

        return model
    def __init__(self, input_shape, train=False):
        """
        
        """
        self.model = self.build(input_shape=input_shape, train=train)

    def summary(self):
        self.model.summary()

    def compile(self, loss, optimizer, metrics):
        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    def to_json(self):
        return self.model.to_json()

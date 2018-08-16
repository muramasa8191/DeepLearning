from tensorflow.python.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D

def conv_block(kernel_size, filters, stage, block, strides=(2, 2), use_bias=True):
    """ Convolution Block
        kernel_size: size of the kernel
        filters: [filter1, filter2, filter3]
        stage: stage number for Layer name
        block: block name for Layer name
    """
    def f(input_tensor):
        """
        main part that can be built from the parameter
        """
        filter1, filter2, filter3 = filters
        cnv_name = 'res{}{}_branch'.format(stage, block)
        bn_name = 'bn{}{}_branch'.format(stage, block)
        
        x = Conv2D(filter1, (1, 1), strides=strides, name=cnv_name + '2a', use_bias=use_bias)(input_tensor)
        x = BatchNormalization(name=bn_name + '2a')(x)
        x = Activation('relu')(x)
        
        x = Conv2D(filter2, (kernel_size, kernel_size), padding='same', name=cnv_name + '2b', use_bias=use_bias)(x)
        x = BatchNormalization(name=bn_name + '2b')(x)
        x = Activation('relu')(x)
        
        x = Conv2D(filter3, (1, 1), name=cnv_name + '2c', use_bias=use_bias)(x)
        x = BatchNormalization(name=bn_name + '2c')(x)
        
        skip = Conv2D(filter3, (1, 1), strides=strides, name=cnv_name + '1', use_bias=use_bias)(x)
        skip = BatchNormalization(name=bn_name + '1')(skip)
        
        x = Add()([x, skip])
        x = Activation('relu', name='res{}{}_out'.format(stage, block))(x)

        return x

    return f

def identity_block(kernel_size, filters, stage, block, use_bias=True):
    """
        
    """
    def f(input_tensor):
        filter1, filter2, filter3 = filters
        cnv_name = 'res{}{}_branch'.format(stage, block)
        bn_name = 'bn{}{}_branch'.format(stage, block)
        
        x = Conv2D(filter1, (1, 1), name=cnv_name + '2a', use_bias=use_bias)(input_tensor)
        x = BatchNormalization(name=bn_name + '2a')(x)
        x = Activation('relu')(x)
        
        x = Conv2D(filter2, (kernel_size, kernel_size), padding='same', name=cnv_name + '2b', use_bias=use_bias)(x)
        x = BatchNormalization(name=bn_name + '2b')(x)
        x = Activation('relu')(x)
        
        x = Conv2D(filter3, (1, 1), name=cnv_name + '2c', use_bias=use_bias)(x)
        x = BatchNormalization(name=bn_name + '2c')(x)
        
        x = Add()([x, in_tensor])
        x = Activation('relu', name='res{}{}_out'.format(stage, block))(x)
        
        return x

    return f

def build_resnet(input_image, num_layer, num_class):
    """
        build ResNet{50, 101}
    """
    assert num_layer in [50, 101]
    
    # Stage 1
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name="conv1")(input_image)
    x = BatchNormalization(name="bn_conv1")(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    
    # Stage 2
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))(x)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')(x)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')(x)
        
    # Stage 3
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')(x)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')(x)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')(x)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')(x)

    # Stage 4
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')(x)
    # accumulate layers depending on the archtecture
    for i in range({50: 5, 101 : 22}[num_layer]):
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98 + i))(x)

    # Stage 5
    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')(x)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')(x)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')(x)

    return x

class ResNet():

    def summary():
    """
    Print summary of the model
    """
        self.model.summary()

class FCNResNet101(ResNet):

    def __init__(self, input_image, num_class):
        """
        Constructor
        """
        self.model = self.build(input_image, num_class)

    def build(self, input_image, num_class):
        """
        Build ResNet101 base FCN model    
        """
        x = build_resnet(101)
        # add classifier
        x = Conv2D(num_class, (1, 1), kernel_initializer='he_normal', activation='linear', padding='valid', strides=(1, 1), kernel_regularizer=l2(weight_decay))(x)
        
        


from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, Concatenate, Activation, Add, BatchNormalization
from tensorflow.python.keras.regularizers import l2

def UNet(input_shape=(224, 224, 3), weight_decay=0., classes=21):
    """ build Unet Model
        Parameters
        ----------
        input_shape : (N, M), the shape for the InputLayer
        weight_decay: weight for L2 regularization
        Reterns
        --------
        UNet model
    """
    input_image = Input(shape=input_shape, name="input_1")

    # enc 1
    x = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), activation='relu', name="enc1_conv1")(input_image)
    x = conv1 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), activation='relu', name="enc1_conv2")(x)
    x = MaxPooling2D(strides=(2, 2), name="block1_pool")(x)
    # enc 2
    x = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), activation='relu', name="enc2_conv1")(x)
    x = conv2 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), activation='relu', name="enc2_conv2")(x)
    x = MaxPooling2D(strides=(2, 2), name="block2_pool")(x)
    # enc 3
    x = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), activation='relu', name="enc3_conv1")(x)
    x = conv3 = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), activation='relu', name="enc3_conv2")(x)
    x = MaxPooling2D(strides=(2, 2), name="block3_pool")(x)
    # enc 4
    x = Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), activation='relu', name="enc4_conv1")(x)
    x = conv4 = Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), activation='relu', name="enc4_conv2")(x)
    x = MaxPooling2D(strides=(2, 2), name="block4_pool")(x)
    # enc 5
    x = Conv2D(1024, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), activation='relu', name="enc5_conv1")(x)
    x = Conv2D(1024, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), activation='relu', name="enc5_conv2")(x)
    x = Conv2DTranspose(512, (2, 2), strides=(2, 2), kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), padding='same', activation='relu', name="block5_deconv")(x)
    # dec 1
    x = Concatenate(axis=-1, name="dec1_concat")([x, conv4])
    x = Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), activation='relu', name="dec1_conv1")(x)
    x = Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), activation='relu', name="dec1_conv2")(x)
    x = Conv2DTranspose(256, (2, 2), strides=(2, 2), kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), padding='same', activation='relu', name="dec1_deconv")(x)
    # dec 2
    x = Concatenate(axis=-1, name="dec2_concat")([x, conv3])
    x = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), activation='relu', name="dec2_conv1")(x)
    x = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), activation='relu', name="dec2_conv2")(x)
    x = Conv2DTranspose(128, (2, 2), strides=(2, 2), kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), padding='same', activation='relu', name="dec2_deconv")(x)
    # dec 3
    x = Concatenate(axis=-1, name="dec3_concat")([x, conv2])
    x = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), activation='relu', name="dec3_conv1")(x)
    x = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), activation='relu', name="dec3_conv2")(x)
    x = Conv2DTranspose(64, (2, 2), strides=(2, 2), kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), padding='same', activation='relu', name="dec3_deconv")(x)
    # dec 4
    x = Concatenate(axis=-1, name="dec4_concat")([x, conv1])
    x = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), activation='relu', name="dec4_conv1")(x)
    x = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), activation='relu', name="dec4_conv2")(x)
    x = Conv2D(classes+1, (1, 1), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), name="dec4_conv3")(x)

    output = Activation('softmax', name="output")(x)
    
    return Model(input_image, output)



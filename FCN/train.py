import os
import math
import numpy as np
from Models.VGG16 import FCN_VGG16
import keras.backend as K
import tensorflow as tf
from Utils.pascal_util import *
from keras.metrics import binary_crossentropy
from tensorflow.python.keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

GPU_COUNT = 1
RESUME = False

""" void should be removed
def crossentropy_without_void(y_true, y_pred):
    y_pred = K.reshape(y_pred, (-1, K.int_shape(y_pred)[-1]))
    log_softmax = tf.nn.log_softmax(y_pred)
    
    y_true = K.one_hot(tf.to_int32(K.flatten(y_true)), K.int_shape(y_pred)[-1]+1)
    unpacked = tf.unstack(y_true, axis=-1)
    y_true = tf.stack(unpacked[:-1], axis=-1)
    
    cross_entropy = -K.sum(y_true * log_softmax, axis=1)
    cross_entropy_mean = K.mean(cross_entropy)
    
    return cross_entropy_mean

def metrics_without_void(y_true, y_pred):
    nb_classes = K.int_shape(y_pred)[-1]
    y_pred = K.reshape(y_pred, (-1, nb_classes))
    
    y_true = K.one_hot(tf.to_int32(K.flatten(y_true)),
                       nb_classes + 1)
                       unpacked = tf.unstack(y_true, axis=-1)
                       legal_labels = ~tf.cast(unpacked[-1], tf.bool)
                       y_true = tf.stack(unpacked[:-1], axis=-1)
                       
    return K.sum(tf.to_float(legal_labels & K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)))) / K.sum(tf.to_float(legal_labels))
"""
class ImageGenerator():
    def __init__(self):
        pass

    def flow_from_directory(self, data, labels, batch_size, steps_per_epoch):
        data_size = len(data)
        while True:
            for batch_num in range(steps_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                img_data, img_label = pascal_data_generator(
                    data[start_index:end_index], 
                    labels[start_index:end_index],
                    size = (224, 224)
                )
                yield img_data, img_label

if __name__ == '__main__':

    batch_size = 32 * GPU_COUNT
    epochs = 100
    lr_base = 0.01 * (float(batch_size) / 16)
    input_shape = (224, 224, 3)
    
    with tf.device("/cpu:0"): 
        model = FCN_VGG16(input_shape, train=True)

    model.compile(
        loss='categorical_crossentropy',
        optimizer = 'adam',#SGD(lr=lr_base, momentum=0.9),
        metrics=['accuracy']
    )

    current_dir = os.path.dirname(os.path.realpath(__file__))
    save_path = os.path.join(current_dir, 'tmp/fcn_vgg16')
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    checkpoint_path = os.path.join(save_path, 'checkpoint_weights.hdf5')
    if RESUME:
        print('checkPoint file:{}'.format(checkpoint_path))
        model.model.load_weights(checkpoint_path, by_name=False)

    model_path = os.path.join(save_path, "model.json")
    # save model structure
    with open(model_path, 'w') as f:
        model_json = model.to_json()
        f.write(model_json)
    
    if GPU_COUNT > 1:
        from keras.utils.training_utils import multi_gpu_model
        model = multi_gpu_model(model, gpus=GPU_COUNT)
    
    def lr_scheduler(epoch):
        lr = lr_base * ((1 - float(epoch)/epochs) ** 0.9)
        print('lr: %f' % lr)
        return lr
    
    scheduler = LearningRateScheduler(lr_scheduler)
    tsb = TensorBoard(log_dir='./logs')

    checkpoint = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True)

    train_files, label_files = get_train_files('Dataset/VOC2012')
#    val_files, val_labels = get_train_files('Dataset/VOC2012')

    steps_per_epoch = int(np.ceil(len(train_files) / float(batch_size)))
#    val_steps = int(np.ceil(len(val_files) / float(batch_size)))

#    gen = ImageGenerator()
#    datagen = ImageDataGenerator()
    img_data, img_labels = pascal_data_generator(train_files, label_files, size=(224, 224))
#    datagen.fit(img_data)
    
#    hist = model.model.fit_generator(
##        generator = gen.flow_from_directory(train_files, label_files, batch_size, steps_per_epoch),
#        generator=datagen.flow(img_data, img_labels, batch_size=batch_size),
#        steps_per_epoch=steps_per_epoch,
#        epochs=epochs,
#        workers=4,
#        use_multiprocessing=True,
##        validation_data = train_data_generator(val_files, val_labels, batch_size, steps_per_epoch), 
##        validation_steps = 
#        callbacks = [tsb, checkpoint]#[scheduler, tsb, checkpoint]
#    )
    hist = model.model.fit(
            img_data, 
            img_labels, 
            initial_epoch=14,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[tsb, checkpoint]
        )
    model.model.save_weights(save_path+'/model.hdf5')

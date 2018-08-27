import os
import math
import numpy as np
from Models.VGG16 import FCN_VGG16
import keras.backend as K
import tensorflow as tf
from utils.pascal_util import *
from keras.metrics import binary_crossentropy
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint, EarlyStopping, TerminateOnNaN
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

GPU_COUNT = 1
RESUME = False

if __name__ == '__main__':

    batch_size = 16 * GPU_COUNT
    epochs = 100
    lr_base = 0.01 * (float(batch_size) / 16)
    input_shape = (224, 224, 3)
    
    with tf.device("/cpu:0"): 
        model = FCN_VGG16(input_shape, train=True, weight_decay=3e-3, classes=22)

    model.compile(
      #        loss=crossentropy_without_ambiguous,
      loss=crossentropy_without_ambiguous,
#      optimizer = 'adam',
      optimizer = SGD(lr=lr_base, momentum=0.9),
      metrics=[categorical_accuracy_without_ambiguous, categorical_accuracy_only_valid_classes]
      )

    model.summary()

    current_dir = os.getcwd()
    save_path = os.path.join(current_dir, 'tmp/fcn_vgg16')
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    checkpoint_path = os.path.join(save_path, 'checkpoint_weights.hdf5')
    if RESUME:
        print('checkPoint file:{}'.format(checkpoint_path))
        model.load_weights(checkpoint_path, by_name=False)

#    model_path = os.path.join(save_path, "model.json")
#    # save model structure
#    with open(model_path, 'w') as f:
#        model_json = model.to_json()
#        f.write(model_json)
    
    if GPU_COUNT > 1:
        from keras.utils.training_utils import multi_gpu_model
        model = multi_gpu_model(model, gpus=GPU_COUNT)
    
    def lr_scheduler(epoch):
        lr = lr_base * ((1 - float(epoch)/epochs) ** 0.9)
        print('lr: %f' % lr)
        return lr
    
    scheduler = LearningRateScheduler(lr_scheduler)
    tsb = TensorBoard(log_dir='./logs')
    early_stopper = EarlyStopping(monitor='loss',
                              min_delta=0.001,
                              patience=30)
    checkpoint = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True)

    train_files, label_files = get_train_files('Dataset/VOC2012')
#    val_files, val_labels = get_train_files('Dataset/VOC2012')

    steps_per_epoch = int(np.ceil(len(train_files) / float(batch_size)))
#    val_steps = int(np.ceil(len(val_files) / float(batch_size)))

    datagen = VocImageDataGenerator(image_shape=input_shape,
        zoom_range=[0.5, 2.0],
        zoom_maintain_shape=True,
        crop_mode='random',
        crop_size=(input_shape[0], input_shape[1]),
        # pad_size=(505, 505),
        rotation_range=0.,
        shear_range=0,
        horizontal_flip=True,
        channel_shift_range=20.,
        fill_mode='constant',
        label_cval=255)
    
    hist = model.fit_generator(
       datagen.flow_from_imageset(
          target_size=(input_shape[0], input_shape[1]),
          directory='Dataset/VOC2012',
          class_mode='categorical',
          classes = 21,
          batch_size=batch_size, 
          shuffle=True,
          loss_shape=None,
          normalize=True,
          ignore_label=255),
       steps_per_epoch=steps_per_epoch,
       epochs=epochs,
       workers=4,
       use_multiprocessing=True,
       callbacks = [tsb, checkpoint, early_stopper, scheduler, TerminateOnNaN()]
    )
    model.save_weights(save_path+'/model.hdf5')

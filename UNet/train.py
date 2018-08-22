import os
import math
import numpy as np
from model import UNet
import keras.backend as K
import tensorflow as tf
from ....pascal_voc_util.pascal_util import *
from keras.metrics import binary_crossentropy
from tensorflow.python.keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint, EarlyStopping, TerminateOnNaN
from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

GPU_COUNT = 1
RESUME = False

if __name__ == '__main__':

    batch_size = 16 * GPU_COUNT
    epochs = 100
    lr_base = 0.01 * (float(batch_size) / 16)
    input_shape = (224, 224, 3)
    
    with tf.device("/cpu:0"): 
        model = UNet(input_shape, weight_decay=3e-3, classes=21)

    model.compile(
        loss='categorical_crossentropy',
        optimizer = 'adam',
#        optimizer = SGD(lr=lr_base, momentum=0.9),
        metrics=['accuracy']
    )

    model.summary()

    current_dir = os.getcwd()
    save_path = os.path.join(current_dir, 'output/unet')
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
                                    featurewise_center=True,
                                    featurewise_std_normalization=True)

    hist = model.fit_generator(
        datagen.flow_from_imageset(
            target_size=(input_shape[0], input_shape[1]),
            directory='Dataset/VOC2012',
            class_mode='categorical',
            classes = 21,
            batch_size = batch_size,
            shuffle=True),
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        workers=4,
        callbacks = [tsb, checkpoint, early_stopper, scheduler, TerminateOnNaN()]
#        use_multiprocessing=True,
   )
    model.save_weights(save_path+'/model.hdf5')

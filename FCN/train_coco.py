import os
import math
import sys
import numpy as np
from Models.VGG16_deconv import *
import keras.backend as K
import tensorflow as tf
from utils.pascal_util import *
from keras.metrics import binary_crossentropy
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint, EarlyStopping, TerminateOnNaN
from keras.optimizers import SGD
from tensorflow.python import debug as tf_debug
from tensorflow.python.debug.lib.debug_data import has_inf_or_nan

GPU_COUNT = 1
RESUME = False

if __name__ == '__main__':

    batch_size = 20 * GPU_COUNT
    epochs = 100
    lr_base = 0.01 * (float(batch_size) / 16)
    input_shape = (224, 224, 3)
    
    with tf.device("/cpu:0"): 
        model = FCN_VGG16(input_shape, classes=183, target=16)

    if not RESUME:
        transferWeight(model, input_shape)
    
    if GPU_COUNT > 1:
        # structure will change after adjusting GPU model
        model.summary()
        from keras.utils.training_utils import multi_gpu_model
        model = multi_gpu_model(model, gpus=GPU_COUNT)

    model.compile(
#        loss=loss_hook,
      loss=crossentropy_without_ambiguous,
#      optimizer = 'adam',
      optimizer = SGD(lr=lr_base, momentum=0.9, clipnorm=1.,decay=5e-4),
      metrics=[categorical_accuracy_without_ambiguous, categorical_accuracy_only_valid_classes]
      )

    model.summary()

    current_dir = os.getcwd()
    save_path = os.path.join(current_dir, 'tmp/fcn_vgg16_coco')
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
    
#    def lr_scheduler(epoch):
#        lr = lr_base * ((1 - float(epoch)/epochs) ** 0.9)
#        print('lr: %f' % lr)
#        return lr
    
#    scheduler = LearningRateScheduler(lr_scheduler)
    tsb = TensorBoard(log_dir='./logs')
    early_stopper = EarlyStopping(monitor='val_loss',
                              min_delta=0.001,
                              patience=30)
    checkpoint = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True)

    train_files, label_files = get_train_files('Dataset/COCO', img_directory='train/', seg_directory='seg/', seg_image_dir='segmentation')
#    val_files, val_labels = get_val_files('Dataset/VOC2012')

    steps_per_epoch = int(np.ceil(len(train_files) / float(batch_size)))
    #    val_steps = int(np.ceil(len(val_files) / float(batch_size)))
    val_steps = 30

    datagen = VocImageDataGenerator(image_shape=input_shape,
        zoom_range=[1.0, 1.0],
        zoom_maintain_shape=True,
        crop_mode='random',
        crop_size=(input_shape[0], input_shape[1]),
        # pad_size=(505, 505),
        rotation_range=0.,
        shear_range=0,
        horizontal_flip=True,
        vertical_flip=True,
        channel_shift_range=20.,
        fill_mode='constant',
        label_cval=0)
    
    # Debug
    if len(sys.argv) > 1 and sys.argv[1] == 1:
        sess = K.get_session()
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        sess.add_tensor_filter('has_inf_or_nan', has_inf_or_nan)
        K.set_session(sess)

    hist = model.fit_generator(
       datagen.flow_from_imageset(
          target_size=(input_shape[0], input_shape[1]),
          directory='Dataset/COCO',
          img_directory='train/', 
          seg_directory='seg/', 
          seg_image_dir='segmentation',
          class_mode='categorical',
          classes = 183,
          batch_size=batch_size, 
          shuffle=True,
          loss_shape=None,
          normalize=True,
          ignore_label=0,
          coco_flg=True),
       steps_per_epoch=steps_per_epoch,
       validation_data=datagen.flow_from_imageset(
          target_size=(input_shape[0], input_shape[1]),
          crop_mode='none',
          directory='Dataset/COCO',
          img_directory='train/', 
          seg_directory='seg/', 
          seg_image_dir='segmentation',
          classes=183,
          batch_size=batch_size, 
          val_flg=True,
          shuffle=True,
          ignore_label=0,
          coco_flg=True),
       validation_steps=val_steps,
       epochs=epochs,
       workers=4,
       use_multiprocessing=True,
       callbacks = [tsb, checkpoint, early_stopper]#, scheduler, TerminateOnNaN()]
    )
    model.save_weights(save_path+'/model.hdf5')

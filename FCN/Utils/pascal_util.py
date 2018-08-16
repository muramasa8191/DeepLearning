
import os
import numpy as np
from PIL import Image

CLASSES = 21
SEGMENTATION_IMAGE_DIR = 'ImageSets/Segmentation/'
TRAIN_LIST_FILE_NAME = 'train.txt'
TRAIN_CLASS_FILE_NAME = 'trainval.txt'
VALIDATION_LIST_FILE_NAME = 'val.txt'

def image_generator(file_paths, size=None, normalization=True):
    """ generate train data and val
    Parameters
    ----------
    file_paths: the arrray of the path to get the image
    size: input size for model. the image will be resized by this size
    normalization: if True, each pixel value will be devided by 255.
    """
    for file_path in file_paths:
        if file_path.endswith(".png") or file_path.endswith(".jpg"):
            # open a image
            image = Image.open(file_path)
            # resize by init_size
            if size is not None and size != image.size:
                image = image.resize(size)
            # delete alpha channel
            if image.mode == "RGBA":
                image = image.convert("RGB")
            image = np.asarray(image)
            if normalization:
                image = image / 255.0
            yield image

def pascal_data_generator(data_paths, val_paths, size=None):
    """ generate data array
    Parameters
    ----------
    train_data_paths: array of the paths for train data
    train_val_paths: array of the paths for train value
    size: size of the image
    Returns
    --------
    
    """
    img_org, img_segmented = [], []
    for image in image_generator(data_paths, size):
        img_org.append(image)
    
    for image in image_generator(val_paths, size, normalization=False):
        img_segmented.append(image)
    
    assert len(img_org) == len(img_segmented)
    
    # Cast to nparray
    img_data = np.asarray(img_org, dtype=np.float32)
    img_segmented = np.asarray(img_segmented, dtype=np.uint8)

    # Cast void pixel to bg
    img_segmented = np.where(img_segmented == 255, 0, img_segmented)

    # change 0 - 21
    identity = np.identity(CLASSES, dtype=np.uint8)
    img_segmented = identity[img_segmented]
    return [img_data, img_segmented] 

def get_train_files(root_dir):
    
    train_data_files = []
    train_class_files = []
    
    path = os.path.join(root_dir, SEGMENTATION_IMAGE_DIR)
    
    img_dir = os.path.join(root_dir + '/', 'JPEGImages/')
    seg_dir = os.path.join(root_dir + '/', 'SegmentationClass/')
    
    with open(os.path.join(path, TRAIN_LIST_FILE_NAME)) as f:
        for s in f:
            file_name = s.rstrip('\r\n')
            train_data_files.append(os.path.join(img_dir, file_name + '.jpg'))
            train_class_files.append(os.path.join(seg_dir, file_name + '.png'))

    return [train_data_files, train_class_files]

def get_val_files(root_dir):
    
    val_data_files = []
    val_class_files = []
    
    path = os.path.join(root_dir, SEGMENTATION_IMAGE_DIR)
    
    img_dir = os.path.join(root_dir + '/', 'JPEGImages/')
    seg_dir = os.path.join(root_dir + '/', 'SegmentationClass/')
    
    with open(os.path.join(path, VALIDATION_LIST_FILE_NAME)) as f:
        for s in f:
            file_name = s.rstrip('\r\n')
            val_data_files.append(os.path.join(img_dir, file_name + '.jpg'))
            val_class_files.append(os.path.join(seg_dir, file_name + '.png'))
    
    return [val_data_files, val_class_files]

def get_class_map():
    """Return the text of each class
        0 - background
        1 to 20 - classes
        255 - void region
        Returns
        -------
        classes_dict : dict
    """
    
    class_names = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
                   'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                   'dog', 'horse', 'motorbike', 'person', 'potted-plant',
                   'sheep', 'sofa', 'train', 'tv/monitor', 'ambigious']
                   
    # dict for class names except for void
    classes_dict = list(enumerate(class_names[:-1]))
    # add void
    classes_dict.append((255, class_names[-1]))
                   
    classes_dict = dict(classes_lut)

    return classes_dict



import sys
import numpy as np
import math
from PIL import Image
from Models.VGG16 import FCN_VGG16
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import preprocess_input

DATA_DIR = 'Dataset/VOC2012/JPEGImages'
LABEL_DIR = 'Dataset/VOC2012/SegmentationClass'

if __name__ == '__main__':
    
    image_path = sys.argv[1].rstrip('\r\n')
    image_shape = (512, 512, 3)
    model = FCN_VGG16(input_shape=image_shape, train=False)
    
    model.summary()
    
    image = Image.open(os.path.join(os.path.dirname(os.path.realpath(__file__)), image_path))
    image = img_to_array(image)
    # normalize
    image = image / 255.

    # padding
    img_height, img_width = image.shape[:2]
    pad_width = max(image_shape[1] - img_width, 0)
    pad_height = max(image_shape[0] - img_height, 0)
    image = np.lib.pad(image, ((math.ceil(pad_height/2), math.floor(pad_height - pad_height/2)), (math.ceil(pad_width/2), math.floor(pad_width - pad_width/2)), (0, 0)), 'constant', constant_values=0.)


    image = np.expand_dims(image, axis=0)
    
    image = preprocess_input(image)
    
    result = model.model.predict(image, batch_size=1)
    result = np.argmax(np.squeeze(result), axis=-1).astype(np.uint8)

    # convert to index color
    result_img = Image.fromarray(result, mode='P')
    
    result_img = result_img.crop((pad_width/2, pad_height/2, pad_width/2+img_width, pad_height/2+img_height))

    result.show(title='result', command=None)

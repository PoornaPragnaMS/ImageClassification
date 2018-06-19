from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import random
import os
from utils import *
from vgg16_classifier import *
from inceptionv3_classifier import *
from resnet50_classifier import *

input_image = (264, 198)

train_data_dir = 'data/264x198'
validation_data_dir = 'data/264x198-val'

nb_class = 23

#######################################
### VGG16
#######################################

epochs = 5
batch_size = 32

model = CustomVGG16('vgg16', input_image, nb_class, weights='vgg16.h5')
model.fit(train_data_dir, validation_data_dir, epochs, batch_size=batch_size)
#vgg16.save('vgg16_weights.h5')
#vgg16.evaluate(validation_data_dir, batch_size)

#######################################


#######################################
### InceptionV3
#######################################

epochs = 5
batch_size = 32

model = CustomInceptionV3('inceptionv3', input_image, nb_class, weights='inceptionv3.h5')
model.fit(train_data_dir, validation_data_dir, epochs, batch_size=batch_size)
#model.save('inceptionv3_weights.h5')
#model.evaluate(validation_data_dir, batch_size)

#######################################


#######################################
### Resnet50
#######################################

epochs = 5
batch_size = 32

model = CustomRestNet50('resnet50', input_image, nb_class, weights='resnet50.h5')
model.fit(train_data_dir, validation_data_dir, epochs, batch_size=batch_size)
#model.save('resnet50_weights.h5')
#model.evaluate(validation_data_dir, batch_size)

#######################################

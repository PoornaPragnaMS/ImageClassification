
# coding: utf-8

# In[2]:

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import optimizers
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.applications.vgg16 import VGG16, preprocess_input
import random
import os
from sklearn.utils import class_weight
import numpy as np

# In[25]:

# dimensions of our images.
img_shape = (200, 150)

train_data_dir = '../data/200x150'
validation_data_dir = '../data/200x150-val'
nb_class = 23
epochs = 100
batch_size = 64


# In[26]:

input_shape = (img_shape[0], img_shape[1], 3)


# In[27]:

def clf_model(input_shape, nb_class):
    model = Sequential()
    model.add(Conv2D(128, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_class))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.Adam(0.0001),
                  metrics=['acc'])
    model.summary()
    
    return model


# In[30]:

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=img_shape,
    shuffle=True,
    batch_size=batch_size,
    class_mode='categorical')


# In[21]:

test_datagen = ImageDataGenerator()

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=img_shape,
    batch_size=batch_size,
    class_mode='categorical')


# In[39]:

nb_train_samples = train_generator.n
nb_validation_samples = validation_generator.n


# In[40]:

model = clf_model(input_shape, nb_class)
if os.path.isfile('cnn_standard.h5'):
    model.load_weights('cnn_standard.h5')

# In[22]:
class_weight = class_weight.compute_class_weight('balanced', 
                                                 np.unique(train_generator.classes),
                                                 train_generator.classes)

checkpointer = ModelCheckpoint('cnn_standard.h5', save_best_only=True, save_weights_only=True)
model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples//batch_size,
    #class_weight=class_weight,
    epochs=epochs,
    callbacks=[checkpointer])

model.save('cnn_standard_final.h5')
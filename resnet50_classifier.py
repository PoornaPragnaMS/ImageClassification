from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model, Model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import optimizers
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.applications import ResNet50
import random
import os
import numpy as np
from utils import *

class CustomRestNet50():
    def __init__(self, prefix, input_shape, nb_class, weights=None, imagenet_weights='imagenet', show_summary=False):
        self.prefix = prefix
        self.input_shape = (input_shape[0], input_shape[1], 3)

        # Initialize 
        self.res50 = ResNet50(weights=imagenet_weights, include_top=False, input_shape=self.input_shape)

        # Adding our new layers 
        x = self.res50.output
        x = Flatten()(x)
        x = Dense(128, activation="relu")(x)
        x = Dropout(0.5)(x)
        x = Dense(128, activation="relu")(x)
        x = Dropout(0.5)(x)
        out = Dense(nb_class, activation="softmax")(x)

        self.model = Model(input = self.res50.input, output = out)

        self.model.compile(loss = "categorical_crossentropy", 
                            optimizer = optimizers.Adam(lr=0.00001),
                            metrics=["accuracy"])
        if show_summary: 
            self.model.summary()
        
        # Load weight if already exist
        if weights and os.path.isfile(weights):
            self.model.load_weights(weights)
            
        print('Compile success', prefix)
        
    def fit(self, train_dir, val_dir, epochs=1, batch_size=128):
        # training data generator
        train_generator = get_image_generator(train_dir, (self.input_shape[0], self.input_shape[1]), batch_size)
        
        # testing data generator
        validation_generator = get_image_generator(val_dir, (self.input_shape[0], self.input_shape[1]), batch_size)
        
        # get number of traing and testing samples
        nb_train_samples = train_generator.n
        nb_validation_samples = validation_generator.n
        
        # create model checkpoint and train
        checkpointer = ModelCheckpoint(self.prefix + '.h5', save_best_only=False, save_weights_only=True)
        self.model.fit_generator(
            train_generator,
            steps_per_epoch=nb_train_samples//batch_size,
            validation_data=validation_generator,
            validation_steps=nb_validation_samples//batch_size,
            #class_weight=class_weight,
            epochs=epochs,
            callbacks=[checkpointer])

        # save the last weights
        self.model.save_weights(self.prefix + '_weights.h5')
        
    def evaluate(self, val_dir, batch_size=128):
        # testing data generator
        validation_generator = get_image_generator(val_dir, (self.input_shape[0], self.input_shape[1]), batch_size, shuffle=False)
        
        # create model checkpoint and train
        result = self.model.evaluate_generator(validation_generator)
        print(result)
        
        return result

    def save(self, file=None):
        if file:
            self.model.save_weights(file)
        else:
            self.model.save_weights(self.prefix + '_final.h5')

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import random
import os
import numpy as np
import pickle
from utils import *
from vgg16_classifier import *
from inceptionv3_classifier import *
from resnet50_classifier import *
from keras.layers import Average
from keras.utils.vis_utils import plot_model

validation_data_dir = 'data/264x198-val'
sub_data_dir = 'data/sub'

batch_size = 32
nb_class = 23

input_shape = {'vgg16': (200, 150), 'resnet50': (264, 198), 'inceptionv3': (200, 150)}

class Ensemble():
    def __init__(self, models, show_summary=False):
        ### load model
        outs = [m.model.outputs[0] for m in models]
        y = Average()(outs)

        self.model = Model([m.model.input for m in models], y, name='ensemble')
        
        self.model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
        
        if show_summary:
            model.summary()
        
        plot_model(self.model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
            
        print('Ensemble model created')
    
    def evaluate(self, val_dir, input_shapes, batch_size=32):
        # testing data generator
        img_shapes = [input_shapes['vgg16'], input_shapes['resnet50'], input_shapes['inceptionv3']]
            
        validation_generator = get_image_generator_by_shapes(val_dir, img_shapes, batch_size=batch_size, shuffle=False)
        
        test_generator = get_image_generator(val_dir, input_shape['vgg16'], batch_size=batch_size, shuffle=False)
        total_sample = test_generator.n
        print('Total test samples:', total_sample)
                
        # create model checkpoint and train
        result = self.model.evaluate_generator(validation_generator,
                                              steps=total_sample//batch_size)
        
        print(result)
        
        return result
    
    
    def predict(self, val_dir, input_shapes, batch_size=1):
        # data generator
        img_shapes = [input_shapes['vgg16'], input_shapes['resnet50'], input_shapes['inceptionv3']]
            
        validation_generator = get_image_generator_by_shapes(val_dir, img_shapes, 
                                                             batch_size=batch_size, shuffle=False)
        
        test_generator = get_image_generator(val_dir, input_shape['vgg16'], batch_size=batch_size, shuffle=False)
        total_sample = test_generator.n
        print('Total test samples:', total_sample)
        
        # get info of data need to predict
        file_names = test_generator.filenames
        y_true = test_generator.classes
        label_map = test_generator.class_indices
        label_map = dict((v,k) for k,v in label_map.items())
                
        # predict class
        y_probas = self.model.predict_generator(validation_generator,
                                              steps=total_sample,
                                              verbose=1)
        y_pred = y_probas.argmax(axis=-1)
        
        print('True label:', y_true[:50])
        print('Predict label:', y_pred[:50])
        
        # save predict
        save_file = {
            'predict_probas' : y_probas,
            'predict_classes' : y_pred,
            'true_classes' : y_true,
            'label_map' : label_map,
            'file_names' : file_names
        }
        with open('ensembel_result.pkl', 'wb') as f:
            pickle.dump(save_file, f)
        
        return y_pred
    
    def save(self, file=None):
        if file:
            self.model.save_weights(file)
        else:
            self.model.save_weights('ensemble_final.h5')
        
    
print('Loading pre-trained models')
list_models = [
    CustomVGG16('vgg16', input_shape['vgg16'], nb_class, weights='vgg16_weights_0.h5', imagenet_weights=None),
    CustomRestNet50('resnet50', input_shape['resnet50'], nb_class, weights='resnet50_weights_0.h5', imagenet_weights=None),
    CustomInceptionV3('inceptionv3', input_shape['inceptionv3'], nb_class, weights='inceptionv3_weights_0.h5', imagenet_weights=None)
]

# load ensemble model based on pre-trained models
model = Ensemble(list_models)

#model.evaluate(validation_data_dir, input_shape, batch_size)

# predict
model.predict(sub_data_dir, input_shape)
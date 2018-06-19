# Setting 01: 
#    - Freeze
#    - Add 1 Dense(256) + Dropout(0.5) + Dense(128) + Dropout(0.2)
#    - Input size: (200,150)
#    - Adam(0.0001)
# --> 100 epochs, 60% acc

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model, load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import optimizers
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.applications import VGG19
import random, os

# ## Transfer: VGG16

# In[9]:

img_shape = (264, 198)

train_data_dir = '../data/264x198'
validation_data_dir = '../data/264x198-val'

epochs = 200
batch_size = 32

input_shape = (img_shape[0], img_shape[1], 3)


# In[10]:

def vgg19_net(input_shape, nb_class):
    # Initialize VGG16 using pre-trained weights on imagenet
    model = VGG19(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # use transfer learning for re-training the last layers
    # Freeze first 25 layers, so that we can retrain 26th and so on using our classes
    #for layer in model.layers:
    #    layer.trainable = False
        
    # Adding our new layers 
    top_layers = model.output
    top_layers = Flatten()(top_layers)
    top_layers = Dense(256, activation="relu")(top_layers)
    top_layers = Dropout(0.5)(top_layers)
    top_layers = Dense(128, activation="relu")(top_layers)
    top_layers = Dropout(0.5)(top_layers)
    out = Dense(nb_class, activation="softmax")(top_layers)
    
    model_final = Model(input = model.input, output = out)
    
    model_final.compile(loss = "categorical_crossentropy", 
                        optimizer = optimizers.Adam(),#optimizers.SGD(lr=0.0001, momentum=0.9), 
                        metrics=["accuracy"])
    model_final.summary()
    
    return model_final 


# In[11]:

train_datagen = ImageDataGenerator(rescale=1./255) #rescale=1. / 255

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=img_shape,
    batch_size=batch_size,
    class_mode='categorical')


# In[ ]:

test_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=img_shape,
    batch_size=batch_size,
    class_mode='categorical')


# In[ ]:

nb_train_samples = train_generator.n
nb_validation_samples = validation_generator.n
nb_class = train_generator.num_classes


# In[ ]:

model = vgg19_net(input_shape, nb_class)
if os.path.isfile('vgg19.h5'):
    model.load_weights('vgg19.h5')

# In[ ]:

checkpointer = ModelCheckpoint('vgg19.h5', save_best_only=False, save_weights_only=True)
model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples//batch_size,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples//batch_size,
    epochs=epochs,
    callbacks=[checkpointer])

model.save('vgg19_final.h5')
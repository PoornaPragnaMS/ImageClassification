
# coding: utf-8

# In[2]:

import pandas as pd
import urllib
import urllib.request as req
import os
import numpy as np
from os import listdir
from PIL import Image
import random
from keras.preprocessing.image import ImageDataGenerator


def download_image(save_path, urls, cats, nb=0, from_idx=0, names=None):
    print('Downloading to', save_path, 'from index', from_idx)
    # create folder to save images
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for cat in np.unique(cats):
        try:
            if not os.path.exists(save_path + cat):
                os.makedirs(save_path + cat)
        except OSError:
            print('Error while creating directory of', save_path)
    
    # download image and save into its folder
    number_download = len(urls[from_idx:]) if nb == 0 else min(nb, len(urls[from_idx:]))
    count = 0
    existed = 0
    skip = 0
    if names == None:
        names = range(number_download)
    for i, name in enumerate(names):
        file_name = save_path + cats[i + from_idx] + '/' + str(name + from_idx) + '.jpg'
        if not os.path.isfile(file_name):
            try:
                req.urlretrieve(urls[i + from_idx], file_name)
                count += 1
                print("Downloaded ", from_idx + existed + count, '-', cats[i])
            except Exception as e:
                print('Error while download from', urls[i + from_idx], e)
                skip += 1
        else:
            existed += 1
    print('Download', count)
    print('Existed', existed)
    print('Skip', skip)

def clean_invalid_img(img_dir):
    bad_file = 0
    for filename in listdir(img_dir):
        try:
            img = Image.open(img_dir + filename)
            img.verify()
        except (IOError, SyntaxError) as e:
            print('Bad file:', filename)
            os.remove(img_dir + filename)
            bad_file +=1
    return bad_file

def clean_dir(dir_image):
    # Clean training data
    print('Cleaning data in', dir_image)
    for cl in listdir(dir_image):
        bad = clean_invalid_img(dir_image + cl + '/')
        if bad > 0: print('Cleaned', bad, 'images in ', cl)

def summary_data(folder):
    data_dic = {}
    for cl in listdir(folder):
        files = listdir(folder + cl)
        data_dic[cl] = len(files)
    for k, v in data_dic.items():
        print('%30s ' % k, v)
    return data_dic

def create_data(data_path, des_path, name='dev', min_val=20, split=0.1, des_size=(264,198)):
    print('Creating train/val data from', data_path, 'to', des_path)
    
    train_path = des_path + name + '/'
    val_path = des_path + name + '-val/'
    
    # create folder at destination dir
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(val_path):
        os.makedirs(val_path)
        
    for cl_folder in os.listdir(data_path):
        print('Process data in', cl_folder)
        # create class folder at destination dir
        if not os.path.exists(train_path + cl_folder):
            os.makedirs(train_path + cl_folder)
        if not os.path.exists(val_path + cl_folder):
            os.makedirs(val_path + cl_folder)
        
        images = os.listdir(data_path + cl_folder)
        if len(images) == 0: continue
        random.seed(91)
        random.shuffle(images)
        
        val_nb = max(int(len(images)*split), min_val)
        
        count = 0
        for img in images[:-val_nb]:
            im_file = data_path + cl_folder + '/' + img
            try :
                im = Image.open(im_file)
                im.thumbnail(des_size, Image.ANTIALIAS)
                im.save(train_path + cl_folder + '/' + img, "JPEG")
                count += 1
            except IOError:
                print("Cannot read image for ", im_file)
                
        for img in images[-val_nb:]:             
            im_file = data_path + cl_folder + '/' + img
            try :
                im = Image.open(im_file)
                im.thumbnail(des_size, Image.ANTIALIAS)
                im.save(val_path + cl_folder + '/' + img, "JPEG")
            except IOError:
                print("Cannot read image for ", im_file)
        
        print('Split train/val:', count, '\t-\t', val_nb)

def get_image_generator(from_dir, img_shape, batch_size, shuffle=True):
    datagen = ImageDataGenerator() #rescale=1. / 255

    generator = datagen.flow_from_directory(
        from_dir,
        shuffle=shuffle,
        target_size=img_shape,
        batch_size=batch_size,
        class_mode='categorical')
    
    return generator

def generate_generator_multiple(generator, nb=3):
    while True:
            X = generator.next()
            yield ([X[0] for _ in range(nb)], X[1])
            
def get_image_generator_by_shapes(from_dir, img_shapes, batch_size, shuffle=True):
    generators = []
    for shape in img_shapes:
        generators.append(get_image_generator(from_dir, shape, batch_size=batch_size, shuffle=shuffle))
    
    while True:
            X = [g.next() for g in generators]
            yield ([x[0] for x in X], X[0][1])
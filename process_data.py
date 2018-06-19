import pandas as pd
import urllib
import urllib.request as req
import os
import numpy as np
from os import listdir
from PIL import Image
import random
from utils import *

folder = 'data/train/'
test_folder = 'data/'

### Download all training images
def download_train_data(folder, from_idx=60000, nb_download=70000):
    ### Read data links, drop null
    df_train = pd.read_csv('data/myntra_train_dataset.csv')
    df_train = df_train[~df_train.Link_to_the_image.isnull()]
    #print(df_train.info())

    image_urls = df_train['Link_to_the_image'].values
    subs = df_train['Sub_category'].values

    ### Download images in to folder
    download_image(folder, image_urls, subs, nb_download, from_idx)

    ### Clean invalid images
    clean_dir(folder)
    
### Download all test images
def download_test_data(folder):
    ### Read data links, drop null
    df_test = pd.read_csv('data/myntra_test.csv')
    df_test = df_test[~df_test.Link_to_the_image.isnull()]

    image_urls = df_test['Link_to_the_image'].values
    subs = ['test'] * len(image_urls)
    names = list(df_test.index)

    ### Download images in to folder
    download_image(folder, image_urls, subs, names=names)

    ### Clean invalid images
    clean_invalid_img(folder + 'test/')

### Download
download_train_data(folder)

### Download test data
download_test_data(test_folder)

### Resize images and split into train/validate set: 264x198, 264x198-val
create_data('data/train/', 'data/', name='264x198', min_val=20, split=0.1, des_size=(264,198))
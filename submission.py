from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import random
import os
from utils import *
from sklearn.metrics import accuracy_score
import pickle

### Read prediction result
result = {}
with open('ensembel_result.pkl', 'rb') as f:
    result = pickle.load(f)
print(result.keys())

### Read submission file
df_sub = pd.read_csv('data/Submission_online.csv')
print(df_sub.info())

### Adding prediction class
names_idx = result['file_names']

with open('class_map', 'rb') as f:
    label_map = pickle.load( f)

predict_classes = result['predict_classes']

for i, cl in zip(names_idx, predict_classes):
    i = i.split('/')[1].split('.')[0]
    df_sub['Sub_category'][int(i)] = label_map[cl]

print(df_sub.head())

### Saving
df_sub.to_csv('data/Submission_online_completed.csv')
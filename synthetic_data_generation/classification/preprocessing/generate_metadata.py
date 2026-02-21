# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 13:17:18 2025

@author: hussain
"""

import os
import pandas as pd
import random

data_dir = '../../../../../data/rare_disease/scabies_with_bg_jpg/'
data = []
for class_name in os.listdir(data_dir):
    folder_path = os.path.join(data_dir, class_name)
    append_data_path = os.path.join(data_dir, class_name)
    if os.path.isdir(folder_path):
        label = class_name
        images = [img for img in os.listdir(folder_path) if img.endswith(('.tiff', '.jpg', '.png', '.jpeg'))]
        random.shuffle(images)
        num_folds = 5
        folds = [[] for _ in range(num_folds)]
        for idx, img_name in enumerate(images):
            fold_idx = idx % num_folds
            folds[fold_idx].append(img_name)
            
        for fold_idx in range(num_folds):
            set_values = ['test' if i == fold_idx else 'train' for i in range(num_folds)]
            for img_name in folds[fold_idx]:
                img_path = os.path.join(folder_path, img_name)
                data.append([img_path, label, 'scabies'] + set_values)
                
df = pd.DataFrame(data, columns=['image', 'label', 'dataset', 'fold0', 'fold1', 'fold2', 'fold3', 'fold4'])   
# csv_file = 'scabies_real_metadata.csv'  
# df.to_csv(csv_file, index=False)
# print(f"CSV file created with exit: {csv_file}")
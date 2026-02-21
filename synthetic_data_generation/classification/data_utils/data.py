# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 12:50:55 2025

@author: hussain
"""

import os
from os.path import expanduser
from os.path import join as ospj
import json
import pickle
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision as tv
from collections import defaultdict
import copy
import h5py
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms as T
import numpy as np
from PIL import Image
import pandas as pd
import random
import torchvision.transforms as tfm
from imageio import imread
from skimage.color import rgb2hsv, hsv2rgb
from .augmenter import HedLighterColorAugmenter, HedLightColorAugmenter, HedStrongColorAugmenter
import os

# from dataset import DatasetScabies, labels_map, T
# from dataset_mix import DatasetScabiesMix

from .utils import (make_dirs, SUBSET_NAMES, configure_metadata, get_image_ids, get_class_labels, GaussianBlur, Solarization)

dataset_image_size = {  
    "scabies": 224,
    "Ace_20":250,   #250,
    "matek":345,   #345, 
    "MLL_20":288,   #288,
    "BMC_22":250,   #288,
    }

labels_map = {
        'HEALTHY': 0,
        'SCAB': 1
    }

NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)
CLIP_NORM_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_NORM_STD = (0.26862954, 0.26130258, 0.27577711)

def get_transforms(model_type):
    if model_type == "clip":
        norm_mean = CLIP_NORM_MEAN
        norm_std = CLIP_NORM_STD
        image_size = 224  # CLIP typically uses 224x224 images

        train_transform = T.Compose([
            T.RandomResizedCrop(size=image_size, scale=(0.8, 1.0), ratio=(0.8, 1.2)),
            T.RandomHorizontalFlip(0.5),
            T.RandomVerticalFlip(0.5),
            T.RandomApply([T.RandomRotation((0, 180))], p=0.33),
            T.RandomApply([T.ColorJitter(brightness=0.5, contrast=0, saturation=1, hue=0.3)], p=0.33),
            T.RandomApply([T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1))], p=0.33),
            T.RandomApply([T.RandomAdjustSharpness(sharpness_factor=0.8)], p=0.33),
            T.ToTensor(),
            T.Normalize(mean=norm_mean, std=norm_std)
        ])

        # Test & validation transformations
        test_transform = T.Compose([
            T.Resize(image_size),  # same as training
            T.ToTensor(),
            T.Normalize(mean=norm_mean, std=norm_std)
        ]) 

        return train_transform, test_transform


    
    elif model_type == "resnet50":
        norm_mean = NORM_MEAN
        norm_std = NORM_STD

        # Train transformations from dataset_wbc.py)
        train_transform = T.Compose([
            T.RandomResizedCrop(size=384, scale=(0.8, 1.0), ratio=(0.8, 1.2)),
            T.RandomHorizontalFlip(0.5),
            T.RandomVerticalFlip(0.5),
            T.RandomApply([T.RandomRotation((0, 180))], p=0.33),
            T.RandomApply([T.ColorJitter(brightness=0.5, contrast=0, saturation=1, hue=0.3)], p=0.33),
            T.RandomApply([T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1))], p=0.33),
            T.RandomApply([T.RandomAdjustSharpness(sharpness_factor=0.8)], p=0.33),
            T.ToTensor(),
            T.Normalize(mean=norm_mean, std=norm_std)
        ])

        # Test & validation transformations
        test_transform = T.Compose([
            T.Resize(384),  # same as training
            T.ToTensor(),
            T.Normalize(mean=norm_mean, std=norm_std)
        ]) 

        return train_transform, test_transform

class DatasetScabies(Dataset):  # bo
    def __init__(self, 
                 dataroot,
                 dataset_selection,
                 labels_map,
                 fold,
                 transform=None,
                 state='train',
                 is_hsv=False,
                 is_hed=False):
        
        super(DatasetScabies, self).__init__()
        
        self.dataroot = os.path.join(dataroot, '')  

        metadata_path = os.path.join(self.dataroot, 'scabies_real_metadata.csv')
        try:
            metadata = pd.read_csv(metadata_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"No CSV file exists at: {metadata_path}")

        set_fold = "fold" + str(fold)  # use 'set' or 'kfold' as per csv file
        if isinstance(dataset_selection, list):
            dataset_index = metadata.dataset.isin(dataset_selection)
        else:
            dataset_index = metadata["dataset"] == dataset_selection
        print(f"Total rows ({dataset_selection}): {dataset_index.sum()}")

        # Filter by fold
        if state == 'train':
            dataset_index = dataset_index & metadata[set_fold].isin(["train"])
        elif state == 'validation':
            dataset_index = dataset_index & metadata[set_fold].isin(["val"])
        elif state == 'test':
            dataset_index = dataset_index & metadata[set_fold].isin(["test"])
        else:
            raise ValueError(f"Unknow state: {state}")
        print(f"Total rows in ({set_fold}, {state}): {dataset_index.sum()}")

        dataset_index = dataset_index[dataset_index].index
        metadata = metadata.loc[dataset_index, :]
        self.metadata = metadata.copy().reset_index(drop=True)
        self.labels_map = labels_map
        self.transform = transform
        self.is_hsv = is_hsv and random.random() < 0.33
        self.is_hed = is_hed and random.random() < 0.33
        
        self.hed_aug = HedLighterColorAugmenter()
        
        # numpy --> tensor
        self.to_tensor = tfm.ToTensor()
        # tensor --> PIL image
        self.from_tensor = tfm.ToPILImage()

    def __len__(self):
        return len(self.metadata)
    
    def read_img(self, path):
        img = Image.open(path)
        if img.mode == 'CMYK':
            img = img.convert('RGB')    
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        return img
    
    def colorize(self, image):
        """ Add color of the given hue to an RGB image.
    
        By default, set the saturation to 1 so that the colors pop!
        """
        hue = random.choice(np.linspace(-0.1, 0.1))
        saturation = random.choice(np.linspace(-1, 1))
        
        # hue = random.choice(np.linspace(0, 1))
        # saturation = random.choice(np.linspace(0, 1))
        # print(f"Valor de hue generado en colorize: {hue}")
        # print(f"Valor de saturation generado en colorize: {saturation}")
        hsv = rgb2hsv(image)
        hsv[:, :, 1] = saturation
        hsv[:, :, 0] = hue
        return hsv2rgb(hsv)            

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        ## get image and label
        dataset =  self.metadata.loc[idx,"dataset"]
        crop_size = dataset_image_size[dataset]
        #print('crop_size:', crop_size)
        
        file_path = self.metadata.loc[idx,"image"]
        #file_path = os.path.join('../../', self.metadata.loc[idx,"image"])
        #image= self.read_img(file_path)
        image= imread(file_path)[:,:,[0,1,2]]
        h1 = (image.shape[0] - crop_size) /2
        h1 = int(h1)
        h2 = (image.shape[0] + crop_size) /2
        h2 = int(h2)
        
        w1 = (image.shape[1] - crop_size) /2
        w1 = int(w1)
        w2 = (image.shape[1] + crop_size) /2
        w2 = int(w2)
        image = image[h1:h2,w1:w2, :]
        
        label_name = self.metadata.loc[idx,"label"]
        # print(f"Etiqueta obtenida: {label_name} (tipo: {type(label_name)})")
        label = self.labels_map[label_name]
        
        if self.is_hsv:
            image = self.colorize(image).clip(0.,1.)
            #print('img hsv:', image.shape, image.min(), image.max())
        
        if self.is_hed:
            self.hed_aug.randomize()
            image = self.hed_aug.transform(image)
            #print('img hed:', image.shape, image.min(), image.max())
        
        img = self.to_tensor(copy.deepcopy(image))
        #print('img tensor:', img.shape, img.min(), img.max())
        image = self.from_tensor(img)
        #print('img PIL:', image.size)
        
        if self.transform:
            image = self.transform(image)
            # raw_image = self.transform(raw_image)
        
        label = torch.tensor(label).long()
        
        return image, label
        #return {'img': image, 'label': label, 'label_name': label_name, 'path': file_path}
        
class DatasetScabiesMix(Dataset):  # bo
    def __init__(self, 
                 dataroot,
                 dataset_selection,
                 labels_map,
                 fold,
                 transform=None,
                 state='train',
                 is_hsv=False,
                 is_hed=False):
        super(DatasetScabiesMix, self).__init__()
        
        self.dataroot = os.path.join(dataroot, '')  

        metadata_path = os.path.join(self.dataroot, 'matek_metadata.csv')
        try:
            metadata = pd.read_csv(metadata_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"No hi ha cap csv file a: {metadata_path}")

        set_fold = "kfold" + str(fold)  # Adaptation for the csv file
        if isinstance(dataset_selection, list):
            dataset_index = metadata.dataset.isin(dataset_selection)
        else:
            dataset_index = metadata["dataset"] == dataset_selection
        print(f"Filas que hi ha en total ({dataset_selection}): {dataset_index.sum()}")

        # Filter by fold
        if state == 'train':
            dataset_index = dataset_index & metadata[set_fold].isin(["train"])
        elif state == 'validation':
            dataset_index = dataset_index & metadata[set_fold].isin(["val"])
        elif state == 'test':
            dataset_index = dataset_index & metadata[set_fold].isin(["test"])
        else:
            raise ValueError(f"Estado desconegut: {state}")
        print(f"Filas després de filtrar per fold ({set_fold}, {state}): {dataset_index.sum()}")

        dataset_index = dataset_index[dataset_index].index
        metadata = metadata.loc[dataset_index, :]
        self.metadata = metadata.copy().reset_index(drop=True)
        self.labels_map = labels_map
        self.transform = transform
        self.is_hsv = is_hsv and random.random() < 0.33
        self.is_hed = is_hed and random.random() < 0.33
        
        self.hed_aug = HedLighterColorAugmenter()
        
        # numpy --> tensor
        self.to_tensor = tfm.ToTensor()
        # tensor --> PIL image
        self.from_tensor = tfm.ToPILImage()

    def __len__(self):
        return len(self.metadata)
    
    def read_img(self, path):
        img = Image.open(path)
        if img.mode == 'CMYK':
            img = img.convert('RGB')    
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        return img
    
    def colorize(self, image):
        """ Add color of the given hue to an RGB image.
    
        By default, set the saturation to 1 so that the colors pop!
        """
        hue = random.choice(np.linspace(-0.1, 0.1))
        saturation = random.choice(np.linspace(-1, 1))
        
        hsv = rgb2hsv(image)
        hsv[:, :, 1] = saturation
        hsv[:, :, 0] = hue
        return hsv2rgb(hsv)            

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        ## get image and label
        dataset =  self.metadata.loc[idx,"dataset"]
        crop_size = dataset_image_size[dataset]
        
        file_path = self.metadata.loc[idx,"image"]
        image = imread(file_path)[:,:,[0,1,2]]
        h1 = (image.shape[0] - crop_size) /2
        h1 = int(h1)
        h2 = (image.shape[0] + crop_size) /2
        h2 = int(h2)
        
        w1 = (image.shape[1] - crop_size) /2
        w1 = int(w1)
        w2 = (image.shape[1] + crop_size) /2
        w2 = int(w2)
        image = image[h1:h2,w1:w2, :]
        
        label_name = self.metadata.loc[idx,"label"]
        label = self.labels_map[label_name]
        
        if self.is_hsv:
            image = self.colorize(image).clip(0.,1.)
        
        if self.is_hed:
            self.hed_aug.randomize()
            image = self.hed_aug.transform(image)
        
        img = self.to_tensor(copy.deepcopy(image))
        image = self.from_tensor(img)
        
        if self.transform:
            image = self.transform(image)
        
        label = torch.tensor(label).long()

        # Leer la columna is_real
        is_real = self.metadata.loc[idx, "is_real"]
        is_real = torch.tensor(is_real).long()

        return image, label, is_real        

def get_data_loader(
    dataroot,  
    dataset_selection="scabies",  
    bs=64, 
    eval_bs=32, #1
    is_rand_aug=True,
    model_type=None,
    fold=0,  
    is_hsv=True,  
    is_hed=True,  
):

    train_transform, test_transform = get_transforms(model_type)

    train_dataset = DatasetScabies(
        dataroot=dataroot,
        dataset_selection=dataset_selection,
        labels_map=labels_map,
        fold=fold,
        transform=train_transform if is_rand_aug else test_transform,
        state='train',
        is_hsv=is_hsv,
        is_hed=is_hed,
    )

    # Crear el DataLoader para entrenamiento
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=bs, 
        shuffle=is_rand_aug,
        prefetch_factor=4, 
        pin_memory=True,
        num_workers=8 #16
    )

    val_dataset = DatasetScabies(
        dataroot=dataroot,
        dataset_selection=dataset_selection,
        labels_map=labels_map,
        fold=fold,
        transform=test_transform,  # same as test
        state='validation', # validation, test
        is_hsv=is_hsv,
        is_hed=is_hed,
    )

    # Crear el DataLoader para validación
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=eval_bs, 
        shuffle=False, 
        num_workers=8,  # 16
        pin_memory=True
    )

    test_dataset = DatasetScabies(
        dataroot=dataroot,
        dataset_selection=dataset_selection,
        labels_map=labels_map,
        fold=fold,
        transform=test_transform,
        state='test',
        is_hsv=is_hsv,
        is_hed=is_hed,
    )

    # Crear el DataLoader para prueba
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=eval_bs, 
        shuffle=False, 
        num_workers=8, #16
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


def get_synth_train_data_loader(
    dataroot,  # Path al CSV que contiene imágenes reales y sintéticas
    dataset_selection="scabies",  # Dataset a seleccionar
    bs=64,
    is_rand_aug=True,
    model_type=None,
    fold=0,  # Fold para k-fold cross-validation
    is_hsv=True,  # Control de HSV
    is_hed=True,  # Control de HED
):
    # Obtener las transformaciones
    train_transform, test_transform = get_transforms(model_type)

    combined_dataset = DatasetScabiesMix(
        dataroot=dataroot,  # Path al CSV
        dataset_selection=dataset_selection,  # Dataset a seleccionar
        labels_map=labels_map,  # Mapeo de etiquetas
        fold=fold,  # Fold para k-fold cross-validation
        transform=train_transform if is_rand_aug else test_transform,  # Transformaciones
        state="train",  # Estado del dataset (entrenamiento)
        is_hsv=is_hsv,  # Transformaciones HSV
        is_hed=is_hed,  # Transformaciones HED
    )

    # Crear el DataLoader para el dataset combinado
    train_loader = torch.utils.data.DataLoader(
        combined_dataset,
        batch_size=bs,
        shuffle=is_rand_aug,  # Barajar los datos si se aplican transformaciones aleatorias
        num_workers=8,  # Ajustar según los recursos disponibles
        pin_memory=True,  # Optimización para transferencias de memoria
    )

    return train_loader
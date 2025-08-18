# -*- coding: utf-8 -*-
"""
Created on Sat Apr 12 17:02:50 2025

@author: hussain
"""

import configargparse
import os
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
import torchvision
import torch.nn as nn
import utils
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR

if __name__ == "__main__":
    p = configargparse.ArgumentParser()
    p.add_argument('--dataset_train', type=str, default='../../../../data/rare_disease/public/lc25000_lung_colon_image_set/colon_image_sets/', help='Dataset train path.')
    p.add_argument('--limit_train_samples', type=bool, default=False, help='subset of total train set.')
    p.add_argument('--train_samples', type=int, default=700, help='No. of random training samples.')
    p.add_argument('--dataset_val', type=str, default='../../../../data/rare_disease/public/lc25000_lung_colon_image_set/colon_val/', help='Dataset validation path.')
    p.add_argument('--dataset_test', type=str, default='../../../../data/rare_disease/public/lc25000_lung_colon_image_set/colon_test/', help='Dataset test path.')
    p.add_argument('--log_dir', type=str, default='output_colon_real', help='Name of the folder to save the model.')
    p.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    p.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate. 1e-3, 1e-4, 5e-5')
    p.add_argument('--num_epochs', type=int, default=50, help='Number of epochs.')
    p.add_argument('--device', type=str, default='gpu', help='Choose the device: "gpu" or "cpu"')
    p.add_argument('--model', type=str, default='maxvit', help='maxvit, swin, efficientnet, resnet50, mobilenetv2, densenet121, resnet101, deit_large, deit_base, volo_d1_224, clip_without_lora , clip_with_lora')
    # clip and lora setting
    p.add_argument('--dataset_name', type=str, default='colon', help='scabies, colon, lung etc.')
    p.add_argument('--clip_download_dir', type=str, default='model/', help='Download clip weights here.')
    p.add_argument('--clip_version', type=str, default='ViT-B/32', help='ViT-B/16, ViT-B/32')
    p.add_argument('--is_lora_image', type=bool, default=True)
    p.add_argument('--is_lora_text', type=bool, default=True)
    opt = p.parse_args()
    
    if opt.device == 'gpu' and torch.cuda.is_available():
        device = torch.device("cuda:0")
        print('Device assigned: GPU (' + torch.cuda.get_device_name(device) + ')\n')
    else:
        device = torch.device("cpu")
        if not torch.cuda.is_available() and opt.device == 'gpu':
            print('GPU not available, device assigned: CPU\n')
        else:
            print('Device assigned: CPU\n')
    
    transformer = utils.get_transformation(opt)
    
    full_train_dataset = torchvision.datasets.ImageFolder(opt.dataset_train, transform=transformer)
    if opt.limit_train_samples:
        targets = full_train_dataset.targets
        splitter = StratifiedShuffleSplit(n_splits=1, train_size=opt.train_samples, random_state=42)
        for train_idx, _ in splitter.split(np.zeros(len(targets)), targets):
            full_train_dataset = Subset(full_train_dataset, train_idx)
            
    train_loader = DataLoader(full_train_dataset, batch_size=opt.batch_size, shuffle=True)
    
    valid_loader = DataLoader(
        torchvision.datasets.ImageFolder(opt.dataset_val, transform=transformer),
        batch_size=opt.batch_size, shuffle=False)
    
    test_loader = DataLoader(
        torchvision.datasets.ImageFolder(opt.dataset_test, transform=transformer),
        batch_size=opt.batch_size, shuffle=False)
    
    os.makedirs(opt.log_dir, exist_ok=True)
    os.makedirs(os.path.join(opt.log_dir, opt.model), exist_ok=True)
    
    model = utils.get_model(opt, device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=opt.learning_rate, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    scheduler = CosineAnnealingLR(optimiser, T_max=opt.num_epochs)
    
    utils.train(opt.model, model, train_loader, valid_loader, loss_fn, optimiser, opt.num_epochs,
          opt.batch_size, opt.learning_rate, device, opt.log_dir, scheduler)
    utils.test(opt.log_dir, opt.model, model, test_loader, device, len(os.listdir(opt.dataset_val)))
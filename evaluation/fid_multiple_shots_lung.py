# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 00:13:54 2025

@author: hussain
"""

import argparse
import torch
from torchvision import transforms
import os
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd

def load_class_images(cls_path, extensions, transform):
    class_images = []
    if os.path.isdir(cls_path):
        for fname in os.listdir(cls_path):
            if fname.lower().endswith(extensions):
                img_path = os.path.join(cls_path, fname)
                img = Image.open(img_path).convert("RGB")
                img = transform(img)
                tensor = pil_to_tensor(img)
                if tensor is not None:
                    class_images.append(tensor)
    return class_images

def load_all_images(root_dir, extensions, transform):
    all_images = []
    for class_name in os.listdir(root_dir):
        cls_path = os.path.join(root_dir, class_name)
        class_images = load_class_images(cls_path, extensions, transform)
        # all_images.append(class_images)
        all_images.extend(class_images)
    return all_images

def update_fid_in_batches(fid, batch_size, tensors, real: bool):
    for i in range(0, len(tensors), batch_size):
        batch = tensors[i:i+batch_size]
        batch_tensor = torch.stack(batch).to(device)
        fid.update(batch_tensor, real=real)
        del batch_tensor
        torch.cuda.empty_cache()
        
def dataset_wise_fid(args, transform):
    real_imgs = load_all_images(args.real_imgs_paths[idx], (args.real_images_extension), transform)
    synthetic_imgs = load_all_images(args.synthetic_imgs_paths[idx], (args.synthetic_images_extension), transform)
    sample_size = min(len(real_imgs), len(synthetic_imgs))
    real_imgs = real_imgs[:sample_size]
    synthetic_imgs = synthetic_imgs[:sample_size]
    fid = FrechetInceptionDistance(feature=2048).to(device)
    update_fid_in_batches(fid, args.batch_size, real_imgs, real=True)
    update_fid_in_batches(fid, args.batch_size, synthetic_imgs, real=False)
    dataset_fid_score = fid.compute().item()
    
    return dataset_fid_score

def class_wise_fid(args, idx, transform):
    classes = sorted(os.listdir(args.real_imgs_paths[idx]))
    class_names, class_fid_score_list = [], []
    for class_name in tqdm(classes, desc="Calculating FID per class..."):
        real_path = os.path.join(args.real_imgs_paths[idx], class_name)
        real_class_imgs = load_class_images(real_path, args.real_images_extension, transform)
        
        synthetic_path = os.path.join(args.synthetic_imgs_paths[idx], class_name)
        synthetic_class_imgs = load_class_images(synthetic_path, args.synthetic_images_extension, transform)
        class_sample_size = min(len(real_class_imgs), len(synthetic_class_imgs))
        
        real_class_imgs = random.sample(real_class_imgs, class_sample_size)
        synthetic_class_imgs = random.sample(synthetic_class_imgs, class_sample_size)
        
        fid_class = FrechetInceptionDistance(feature=2048).to(device)
        update_fid_in_batches(fid_class, args.batch_size, real_class_imgs, real=True)
        update_fid_in_batches(fid_class, args.batch_size, synthetic_class_imgs, real=False)
        class_fid_score = fid_class.compute().item()
        class_names.append(class_name)
        class_fid_score_list.append(class_fid_score)
    return class_names, class_fid_score_list

def draw_graph(csv_file):
    df = pd.read_csv(csv_file)
    shots = df['few_shot'].tolist()
    fid_cls_1 = df['lung_aca'].tolist()
    fid_cls_2 = df['lung_n'].tolist()
    fid_cls_3 = df['lung_scc'].tolist()
    fid_all = df['all'].tolist()

    x = np.arange(len(shots))
    width = 0.20
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width, fid_cls_1, width, label='lung_aca', color='#66C2A5')
    bars2 = ax.bar(x, fid_cls_2, width, label='lung_n', color='#FC8D62')
    bars3 = ax.bar(x + width, fid_cls_3, width, label='lung_scc', color='grey')
    bars4 = ax.bar(x + width + width, fid_all, width, label='All (Dataset)', color='#8DA0CB')
    ax.set_xlabel('Number of Shots (Samples used to generate synthetic data)', fontsize=12)
    ax.set_ylabel('FID Score (lower is better)', fontsize=12)
    # ax.set_title('FID Scores per Class vs Number of Shots', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(shots)
    ax.legend() # ax.legend(title='Class)
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.0f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 4),
                        textcoords="offset points",
                        ha='center', va='bottom')
    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)
    add_labels(bars4)
    # Optional: grid and layout adjustments
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('fid_bar_plot.pdf', dpi=300)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--few_shots_list', metavar='N', type=str, nargs='*', default=[2, 5, 10, 20, 50],
                        help='a list of few-shots from real images')
    parser.add_argument('--real_imgs_paths', metavar='N', type=str, nargs='*',
                        default=['../../../../../data/rare_disease/public/lc25000_lung_colon_image_set/lung_image_sets/',
                                 '../../../../../data/rare_disease/public/lc25000_lung_colon_image_set/lung_image_sets/',
                                 '../../../../../data/rare_disease/public/lc25000_lung_colon_image_set/lung_image_sets/',
                                 '../../../../../data/rare_disease/public/lc25000_lung_colon_image_set/lung_image_sets/',
                                 '../../../../../data/rare_disease/public/lc25000_lung_colon_image_set/lung_image_sets/'],
                        help='list of paths to real images')
    parser.add_argument('--real_images_extension', type=str, default='.jpeg')
    parser.add_argument('--synthetic_imgs_paths', metavar='N', type=str, nargs='*',
                        default=['../generation/generate/output_lung_2_shot/lung/sd2.1/gs2.0_nis50/shot2_seed0_template1_lr0.0001_ep300/train/', 
                                 '../generation/generate/output_lung_5_shot/lung/sd2.1/gs2.0_nis50/shot5_seed0_template1_lr0.0001_ep300/train/',
                                 '../generation/generate/output_lung_10_shot/lung/sd2.1/gs2.0_nis50/shot10_seed0_template1_lr0.0001_ep300/train/',
                                 '../generation/generate/output_lung_20_shot/lung/sd2.1/gs2.0_nis50/shot20_seed0_template1_lr0.0001_ep300/train/',
                                 '../generation/generate/output_lung_50_shot/lung/sd2.1/gs2.0_nis50/shot50_seed0_template1_lr0.0001_ep300/train/'], 
                        help='a list of paths to synthetic images')
    parser.add_argument('--synthetic_images_extension', type=str, default='.png')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--image_crop_size', type=int, default=224)
    parser.add_argument('--output_file_name', type=str, default='multishots_fid.csv')
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transform = transforms.Resize((args.image_crop_size, args.image_crop_size))
    
    shots_list, dataset_fid_list, class_name_list, class_fid_list = [], [], [], []
    
    for idx, shot in enumerate((args.few_shots_list)):
        print('Processing for few shot: ', shot)
        
        dataset_fid = dataset_wise_fid(args, transform)
        class_names, class_fid = class_wise_fid(args, idx, transform)
        
        shots_list.append(shot)
        dataset_fid_list.append(dataset_fid)
        class_name_list.append(class_names)
        class_fid_list.append(class_fid)
        
    header, rows = [], []
    header.append('few_shot')
    for val in class_name_list[0]:
        header.append(val)
    header.append('all')
    
    for index, shot in enumerate((args.few_shots_list)):
        one_row = []
        one_row.append(shot)
        for cls_fid in class_fid_list[index]:
            one_row.append(cls_fid)
        one_row.append(dataset_fid_list[index])
        rows.append(one_row)
        
    with open(args.output_file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)
            
    print(f"CSV file '{args.output_file_name}' created successfully.")
    
    draw_graph(args.output_file_name)
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 23:07:06 2025

@author: hussain
"""

import matplotlib.pyplot as plt

few_shot = [2, 5, 10, 20, 50]
resnet50 = [0.4526, 0.4895, 0.4895, 0.5842, 0.8211]
densenet121 = [0.4842, 0.4684, 0.4947, 0.5737, 0.8000]
maxvit = [0.4263, 0.4947, 0.5316, 0.6684, 0.8316]
clip_16 = [0.4632, 0.5000, 0.5684, 0.6947, 0.8474]
clip_32 = [0.5789, 0.5947, 0.6484, 0.6474, 0.8842]

plt.plot(few_shot, resnet50, label='ResNet-50', color='blue', marker='o')
plt.plot(few_shot, densenet121, label='DenseNet121', color='red', marker='s')
plt.plot(few_shot, maxvit, label='MaxViT', color='green', marker='+')
plt.plot(few_shot, clip_16, label='CLIP (ViT-B/16)', color='orange', marker='*')
plt.plot(few_shot, clip_32, label='CLIP (ViT-B/32)', color='purple', marker='d')

plt.xlabel('Few Shot (Samples to guide the synthetic generation)')
plt.ylabel('Accuracy')
plt.title('')
plt.legend()
plt.savefig('few_shot_classifier.pdf', dpi=300)
plt.show()
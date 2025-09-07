# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 23:07:06 2025

@author: hussain
"""

import matplotlib.pyplot as plt

few_shot = [350, 700, 1000, 2000, 4000]


densenet121 = [0.9167, 0.9195, 0.9243, 0.9289, 0.9367]
deitl = [0.9167, 0.9228, 0.9348, 0.9390, 0.9500]
clip_16 = [0.7667, 0.7890, 0.8084, 0.8374, 0.8600]
clip_32 = [0.7467, 0.7629, 0.7911, 0.8361, 0.8633]

plt.plot(few_shot, densenet121, label='DenseNet121', color='red', marker='s')
plt.plot(few_shot, deitl, label='Deit-L', color='green', marker='+')
plt.plot(few_shot, clip_16, label='CLIP (ViT-B/16)', color='orange', marker='*')
plt.plot(few_shot, clip_32, label='CLIP (ViT-B/32)', color='purple', marker='d')

plt.xlabel('Synthetic training samples')
plt.ylabel('Accuracy')
plt.title('')
plt.legend()
plt.savefig('train_sample_accuracy_colon.pdf', dpi=300)
plt.show()


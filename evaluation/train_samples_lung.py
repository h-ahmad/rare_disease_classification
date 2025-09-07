# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 23:07:06 2025

@author: hussain
"""

import matplotlib.pyplot as plt

few_shot = [350, 700, 1000, 2000, 4000]
maxvit = [0.8911, 0.8948, 0.8973, 0.9017, 0.9044]
deitb = [0.8933, 0.8919, 0.8930, 0.8932, 0.8933]
clip_16 = [0.8578, 0.8745, 0.8824, 0.8994, 0.9067]
clip_32 = [0.8733, 0.8748, 0.8779, 0.8768, 0.8800]

plt.plot(few_shot, maxvit, label='MaxViT', color='green', marker='+')
plt.plot(few_shot, deitb, label='Deit-B', color='red', marker='s')
plt.plot(few_shot, clip_16, label='CLIP (ViT-B/16)', color='orange', marker='*')
plt.plot(few_shot, clip_32, label='CLIP (ViT-B/32)', color='purple', marker='d')

plt.xlabel('Synthetic training samples')
plt.ylabel('Accuracy')
plt.title('')
plt.legend()
plt.savefig('train_sample_accuracy_lung.pdf', dpi=300)
plt.show()


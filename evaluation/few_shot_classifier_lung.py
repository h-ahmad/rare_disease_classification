# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 23:07:06 2025

@author: hussain
"""

import matplotlib.pyplot as plt

few_shot = [2, 5, 10, 20, 50]
maxvit = [0.4589, 0.5378, 0.6845, 0.7865, 0.8911]
deitb = [0.4786, 0.5587, 0.6175, 0.8213, 0.8933]
clip_16 = [0.4231, 0.4456, 0.5806, 0.7148, 0.8578]
clip_32 = [0.4350, 0.4800, 0.5312, 0.7325, 0.8733]

plt.plot(few_shot, maxvit, label='MaxViT', color='red', marker='s')
plt.plot(few_shot, deitb, label='Deit-B', color='blue', marker='o')
plt.plot(few_shot, clip_16, label='CLIP (ViT-B/16)', color='orange', marker='*')
plt.plot(few_shot, clip_32, label='CLIP (ViT-B/32)', color='purple', marker='d')

plt.xlabel('Few Shot (Samples to guide the synthetic generation)')
plt.ylabel('Accuracy')
plt.title('')
plt.legend()
plt.savefig('few_shot_classifier_lung.pdf', dpi=300)
plt.show()
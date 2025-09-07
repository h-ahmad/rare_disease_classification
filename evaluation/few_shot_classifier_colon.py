# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 23:07:06 2025

@author: hussain
"""

import matplotlib.pyplot as plt

few_shot = [2, 5, 10, 20, 50]
densenet121 = [0.5942, 0.5782, 0.6157, 0.6837, 0.9167]
deit_l = [0.6178, 0.6345, 0.7254, 0.8542, 0.9167]
clip_16 = [0.4579, 0.4681, 0.5124, 0.6248, 0.7667]
clip_32 = [0.3997, 0.4365, 0.5465, 0.6978, 0.7467]

plt.plot(few_shot, densenet121, label='DenseNet121', color='red', marker='s')
plt.plot(few_shot, deit_l, label='Deit-L', color='blue', marker='o')
plt.plot(few_shot, clip_16, label='CLIP (ViT-B/16)', color='orange', marker='*')
plt.plot(few_shot, clip_32, label='CLIP (ViT-B/32)', color='purple', marker='d')

plt.xlabel('Few Shot (Samples to guide the synthetic generation)')
plt.ylabel('Accuracy')
plt.title('')
plt.legend()
plt.savefig('few_shot_classifier_colon.pdf', dpi=300)
plt.show()
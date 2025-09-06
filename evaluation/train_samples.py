# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 23:07:06 2025

@author: hussain
"""

import matplotlib.pyplot as plt

few_shot = [100, 200, 300, 400, 500, 600, 700]
resnet50 = [0.6895, 0.7053, 0.7368, 0.7263 , 0.7579, 0.7737, 0.8211]
densenet121 = [0.7263, 0.7421, 0.7526, 0.7421, 0.7632, 0.7632, 0.80]
maxvit = [0.7211, 0.7684, 0.7842, 0.8074, 0.8158, 0.8316, 0.8316]
clip_16 = [0.80, 0.8126, 0.8084, 0.8242, 0.8279, 0.8306, 0.8361]
clip_32 = [0.81, 0.8316, 0.8421, 0.8632, 0.8579, 0.8779, 0.8842]

plt.plot(few_shot, resnet50, label='ResNet-50', color='blue', marker='o')
plt.plot(few_shot, densenet121, label='DenseNet121', color='red', marker='s')
plt.plot(few_shot, maxvit, label='MaxViT', color='green', marker='+')
plt.plot(few_shot, clip_16, label='CLIP (ViT-B/16)', color='orange', marker='*')
plt.plot(few_shot, clip_32, label='CLIP (ViT-B/32)', color='purple', marker='d')

plt.xlabel('Synthetic training samples')
plt.ylabel('Accuracy')
plt.title('')
plt.legend()
plt.savefig('train_sample_accuracy.pdf', dpi=300)
plt.show()


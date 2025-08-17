# -*- coding: utf-8 -*-
"""
Created on Sat Apr 12 20:57:57 2025

@author: hussain
"""

import timm
import torch
import torch.nn as nn

class SwinTransformer(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, model_name='swin_tiny_patch4_window7_224'):
        super(SwinTransformer, self).__init__()
        # Load base model
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)  # no classifier
        in_features = self.backbone.num_features

        # Custom classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        out = self.classifier(features)
        return out
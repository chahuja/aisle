import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pdb

from .layers import *
from .speech2gesture import Speech2Gesture_D

import torch
import torch.nn as nn

class StyleClassifier_G(nn.Module):
  def __init__(self, time_steps=64, in_channels=256, p=0, style_dict={}, **kwargs):
    super().__init__()
    out_feats = len(style_dict)

    self.classifier = nn.ModuleList()
    self.classifier.append(ConvNormRelu(in_channels, 64, downsample=True))
    self.classifier.append(ConvNormRelu(64, 128, downsample=True))
    self.classifier.append(ConvNormRelu(128, 128, downsample=True))
    self.classifier.append(ConvNormRelu(128, 256, downsample=True))
    self.classifier.append(ConvNormRelu(256, 256, downsample=True))
    self.classifier.append(ConvNormRelu(256, out_feats, downsample=True))
    self.model = nn.Sequential(*self.classifier)

  def forward(self, x, y, **kwargs):
    y_cap = self.model(x.transpose(-1, -2)).squeeze(-1)

    #internal_losses = [torch.nn.functional.cross_entropy(y_cap, y)]
    internal_losses = []
    
    return y_cap, internal_losses

class StyleClassifier2_G(nn.Module):
  def __init__(self, time_steps=64, in_channels=256, p=0, style_dict={}, **kwargs):
    super().__init__()
    out_feats = len(style_dict)

    self.classifier = nn.ModuleList()
    self.classifier.append(ConvNormRelu(in_channels, 64, downsample=True)) #96 x 64 -> 64 x 32
    self.classifier.append(ConvNormRelu(64, 64, downsample=False)) #64 x 32-> 64 x 32
    self.classifier.append(ConvNormRelu(64, 64, downsample=False)) #64 x 32-> 64 x 32
    self.classifier.append(nn.MaxPool1d(2)) #64 x 16
    self.classifier.append(ConvNormRelu(64, 128, downsample=False)) #64 x 16 -> 128 x 16
    self.classifier.append(ConvNormRelu(128, 128, downsample=False)) #128 x 16 -> 128 x 16
    self.classifier.append(ConvNormRelu(128, 128, downsample=False)) #128 x 16 -> 128 x 16
    self.classifier.append(nn.MaxPool1d(2)) #128 x 8
    self.classifier.append(ConvNormRelu(128, 256, downsample=False)) #128 x 8 -> 256 x 8
    self.classifier.append(ConvNormRelu(256, 256, downsample=False)) #256 x 8 -> 256 x 8
    self.classifier.append(ConvNormRelu(256, 256, downsample=False)) #256 x 8 -> 256 x 8
    self.classifier.append(nn.MaxPool1d(2)) #256 x 4
    self.classifier.append(ConvNormRelu(256, 256, downsample=True)) #256 x 4 -> 256 x 2
    self.classifier.append(ConvNormRelu(256, out_feats, downsample=True)) #256 x 2 -> 25 x 1
    #self.model = nn.Sequential(*self.classifier)

  def forward(self, x, y, **kwargs):
    x = x.transpose(-1, -2)
    x_queue = []
    for i, model in enumerate(self.classifier[:-2]):
      if i % 4 == 2:
        x = model(x) + x_queue.pop()
      else:
        x = model(x)
      if i % 4 == 0:
        x_queue.append(x)
    x = nn.Sequential(*self.classifier[-2:])(x)
    x = x.squeeze(-1)
    #internal_losses = [torch.nn.functional.cross_entropy(y_cap, y)]
    internal_losses = []
    
    return x, internal_losses

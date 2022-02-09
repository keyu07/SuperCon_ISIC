import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import random
import os
import glob
import sys
import argparse
import logging
import time
import PIL
import torchvision
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch.transforms import ToTensor


class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p
    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)

input_size = 224
augmentation = torch.nn.Sequential(
            RandomApply(
                transforms.ColorJitter(0.8, 0.8, 0.8, 0.2),
                p = 0.3
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
            RandomApply(
                transforms.GaussianBlur((3, 3), (1.0, 2.0)),
                p = 0.2
            ),
            transforms.RandomResizedCrop((input_size, input_size)),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
            std=[0.247, 0.243, 0.261]),
        )

def aug(input_size=224):
    augmentation = transforms.Compose([
      transforms.RandomResizedCrop((input_size,input_size)),
      RandomApply(
          transforms.ColorJitter(0.8, 0.8, 0.8, 0.2),
          p = 0.3
      ),
      transforms.RandomGrayscale(p=0.2),
      transforms.RandomHorizontalFlip(),
      RandomApply(
          transforms.GaussianBlur((3, 3), (1.0, 2.0)),
          p = 0.2
      ),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406],
      std=[0.229, 0.224, 0.225]),])
    return augmentation

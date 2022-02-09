import sys
import collections
import time
import numpy as np
import torch
import random
import os
import shutil
from PIL import Image
import datetime
import torch.optim as optim


def rep_optimizer(netE, rep_head, lr):

    optimizer = optim.SGD(list(netE.parameters())+list(rep_head.parameters()), lr=lr, momentum=0.9, weight_decay=5e-5)

    return optimizer

def downstream_optimizer(classifier, lr):

    optimizer = optim.SGD(classifier.parameters(), lr=lr, momentum=0.9, weight_decay=5e-5)

    return optimizer

def class_optimizer(netE, netC, lr):
    
    optimizer = optim.SGD(list(netE.parameters())+list(netC.parameters()), lr=lr, momentum=0.9, weight_decay=5e-5)
    
    return optimizer
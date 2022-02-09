from torchvision import transforms
from torch.utils.data import Dataset
import torch
import os
from PIL import Image
import pandas as pd
import numpy as np
from augmentation import aug

class SuperconData(Dataset):
    def __init__(self, csv_file, root_dir,transform):

        self.file = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.label_arr = np.asarray(self.file.iloc[:, -1])

        self.transform = transforms.Compose([
                                             transforms.Resize((224, 224)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(([0.485, 0.456, 0.406]), ([0.229, 0.224, 0.225]))
                                             ])
        self.aug = transform


    def __len__(self):
        return len(self.file)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, '{}.jpg'.format(self.file.iloc[idx, 0]))
        image = Image.open(img_name).convert('RGB')
        org_img = self.transform(image)
        pre_img = self.aug(image)
        y = self.label_arr[idx]
        return org_img, pre_img, y

class TrainData(Dataset):
    def __init__(self, csv_file, root_dir,transform):

        self.file = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.label_arr = np.asarray(self.file.iloc[:, -1])
        self.transform = transform


    def __len__(self):
        return len(self.file)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, '{}.jpg'.format(self.file.iloc[idx, 0]))
        org_img = Image.open(img_name).convert('RGB')
        org_img = self.transform(org_img)
        y = self.label_arr[idx]
        return org_img, y

class TestData(Dataset):
    def __init__(self, csv_file, root_dir):

        self.file = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.label_arr = np.asarray(self.file.iloc[:, -1])
        self.transform = transforms.Compose([
                                             transforms.Resize((224, 224)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(([0.485, 0.456, 0.406]), ([0.229, 0.224, 0.225]))
                                             ])
    def __len__(self):
        return len(self.file)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, '{}.jpg'.format(self.file.iloc[idx, 0]))
        image = Image.open(img_name).convert('RGB')
        sample = self.transform(image)
        y = self.label_arr[idx]
        return sample, y

def loaders(mode, train_batchsz=128, test_batchsz=128):
    path_20 = '/path_to_your_image_file'
    if mode == '19+20':
        file = '/path_to_your_listfile/Train19+20.txt'
        # Replace the SuperconData to TrainData when you run Classifier_FineTune.py
        dataset = SuperconData(file, path_20, aug())
        loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=train_batchsz, num_workers=12)
        
    if mode == '2020only':
        file = '/path_to_your_listfile/ISIC2020_train.txt'
        # Replace the SuperconData to TrainData when you run Classifier_FineTune.py
        dataset = SuperconData(file, path_20, aug())
        loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=train_batchsz, num_workers=12)
        
    if mode == 'test':
        file = '/path_to_your_listfile/ISIC2020_test.csv'
        dataset = TestData(file, path_20)
        loader = torch.utils.data.DataLoader(dataset, shuffle=False, batch_size=test_batchsz, num_workers=12)
    return loader

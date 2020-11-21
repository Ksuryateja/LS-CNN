import torchvision.transforms as transforms
import torch.utils.data as data
import numpy as np
import pandas as pd
from PIL import Image
import os
import torch


def img_loader(path):
    try:
        with open(path, 'rb') as f:
            img = Image.open(path)
            return img
    except :
        print(f'Cannot load image {path}')

class  CASIAWebFace(data.Dataset):
    def __init__(self, imgInfo_csv, transform = None, loader = img_loader):

        self.transform = transform
        self.loader = loader

        imgInfo = pd.read_csv(imgInfo_csv)
        img_list = imgInfo['image'].to_list()
        label_list = imgInfo['label'].to_list()

        self.img_list = img_list
        self.label_list = label_list
        self.class_num = len(np.unique(self.label_list))

        print(f'Dataset size : {len(self.img_list)} Classes : {self.class_num}')

    def __getitem__(self, index):
        img_path = self.img_list[index]
        label = self.label_list[index]

        img = self.loader(img_path)
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = torch.from_numpy(img)

        return img, label

    def __len__(self):
        return len(self.img_list)

if __name__ == '__main__':

    transform = transforms.Compose([
                
                transforms.RandomCrop(128),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
                ])
    dataset = CASIAWebFace(imgInfo_csv = './data/img_info.csv', transform = transform)
    trainloader = data.DataLoader(dataset, batch_size = 64, shuffle = True, num_workers= 2, drop_last=False)
    print(len(dataset))




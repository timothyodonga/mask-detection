import csv
import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image
import torchvision


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, root, img_list, transforms=None):
        self.root = root
        self.transforms = transforms
        self.img_list = img_list

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        raise NotImplemented


class TrainDataset(MyDataset):
    def __init__(self, root, img_list, transforms=None):
        super().__init__(root, img_list, transforms)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.img_list[idx][0])
        img = Image.open(img_path).convert("RGB")
        label = self.img_list[idx][1]

        if self.transforms is not None:
            img = self.transforms(img)

        return img, label


class TestDataset(MyDataset):
    def __init__(self, root, img_list, transforms=None):
        super().__init__(root, img_list, transforms)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.img_list[idx][0])
        img = Image.open(img_path).convert("RGB")

        if self.transforms is not None:
            img = self.transforms(img)

        return img


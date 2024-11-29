import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode
from torch.utils.data import Dataset
from PIL import Image

class COCO_dataset(Dataset):
    def __init__(self, images_path, transform=None, image_num=None):
        self.images_path = images_path 
        self.transform = transform  
        self.image_list = os.listdir(images_path)
        if image_num is not None:
            self.image_list = self.image_list[:image_num]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_path = os.path.join(self.images_path, self.image_list[index])
        image = Image.open(image_path)
        transformed_image = self.transform(image)
        
        if self.transform is not None:
            image = self.transform(image)
        return transformed_image

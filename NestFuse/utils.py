import torchvision.transforms.functional as TF
import torch
import os
import numpy as np

from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image


def get_test_images(paths, height=None, width=None):
    ImageToTensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=1),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        image = get_image(path)
        image_np = np.array(image, dtype=np.uint32)
        image = ImageToTensor(image).float().numpy()
    images.append(image)
    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).float()

    return images

def get_image(path):
    image = Image.open(path).convert('RGB')

    return image
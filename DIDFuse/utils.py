import torchvision.transforms.functional as TF
import torch
import os
import numpy as np

from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

class Customdataset(Dataset):
    def __init__(self, transform=None, vis_dataset=None, ir_dataset=None):
        self.vis_path = [f for f in os.listdir(vis_dataset) if os.path.isfile(os.path.join(vis_dataset, f))]
        self.ir_path = [f for f in os.listdir(ir_dataset) if os.path.isfile(os.path.join(ir_dataset, f))]
        
        self.vis_path.sort()
        self.ir_path.sort()
        
        self.vis_dataset_path = vis_dataset
        self.ir_dataset_path = ir_dataset
        
        self.transform = transform
                
    def __getitem__(self, index):
        '''Extract sample datas
        '''
        vis_path = os.path.join(self.vis_dataset_path, self.vis_path[index])
        ir_path = os.path.join(self.ir_dataset_path, self.ir_path[index])
        
        vis_img = Image.open(vis_path)
        ir_img = Image.open(ir_path)
        
        vis_img = self.transform(vis_img)
        ir_img = self.transform(ir_img)
        
        return vis_img, ir_img

    def __len__(self):
        return len(self.vis_path)


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
import argparse
import torch
import utils
import utils_coco
import torch.nn as nn
import sys
import datetime
import time
import os

from fusion_strategy import fusion
from model import NestFuse
from torch.utils.data import DataLoader
from torchvision import transforms
from pytorch_msssim import ssim # SSIM
from matplotlib import pyplot as plt
    
def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', type=str, default=r"C:\Users\USER\Desktop\Dataset\MS-COCO\images\train\train2017", help="KAIST Dataset")
    parser.add_argument('--batch_size', type=int, default=4, help="batch size for training")
    parser.add_argument('--epochs', type=int, default=1, help="number of training epochs")
    parser.add_argument('--hyperparamter', type=int, default=100, help="loss of lambda")
    parser.add_argument('--lr', type=float, default=1e-4, help="learning rate")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device: ', device)
    
    trans = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=1),
        transforms.Normalize((0.5,), (0.5))
    ])
    
    dataset = utils_coco.COCO_dataset(images_path=args.dataset, transform=trans, image_num=5000)
    train_dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    print(f'===> Loading Dataset Completed and Length of Dataset: {len(dataset)}')

    fusion_model = nn.DataParallel(NestFuse(), device_ids=[0, 1])
    fusion_model.to(device) 
    optimizer = torch.optim.Adam(fusion_model.parameters(), lr=args.lr)
    
    model_param_path = './saved_models'
    if not os.path.exists(model_param_path):
        os.mkdir(model_param_path)
    
    prev_time = time.time()
    mse_loss = nn.MSELoss()
    for epoch in range(args.epochs):
        fusion_model.train()
        
        for batch, img in enumerate(train_dataloader):
            img = img.to(device)
            
            fusion_output = fusion_model(img)
            L_p = mse_loss(fusion_output, img)  # L_pixel
            L_ssim = ssim(fusion_output, img)
            
            total_loss = L_p + args.hyperparamter * (1-L_ssim)
            
            total_loss.backward()
            optimizer.zero_grad()
            optimizer.step()

            batches_done = epoch * len(train_dataloader) + batch
            batches_left = args.epochs * len(train_dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()
            # Print log
            sys.stdout.write(
                "\rTrain : [Epoch %d/%d] [Batch %d/%d] [L_p: %f] [L_ssim: %f] [Total Loss: %f] ETA: %s"
                % (
                    epoch,
                    args.epochs,
                    batch,
                    len(train_dataloader),
                    L_p.item(),
                    1 - L_ssim.item(),
                    total_loss.item(),
                    time_left,
                )
            )
        
        torch.save(fusion_model.state_dict(), "./saved_models/%s/model_fusion%d.pth" % ("basic", epoch))
    
if __name__ == "__main__":
    main()
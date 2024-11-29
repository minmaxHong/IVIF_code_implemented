import argparse
import torch
import utils
import torch.nn as nn
import sys
import datetime
import time
import os

from model import DenseFuse
from torch.utils.data import DataLoader
from torchvision import transforms
from pytorch_msssim import ssim # SSIM
    
def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--vis_dataset', type=str, default=r"C:\Users\USER\Desktop\Dataset\KAIST", help="KAIST Dataset")
    parser.add_argument('--ir_dataset', type=str, default=r"C:\Users\USER\Desktop\Dataset\KAIST", help="KAIST Dataset")
    parser.add_argument('--batch_size', type=int, default=4, help="batch size for training")
    parser.add_argument('--epochs', type=int, default=20, help="number of training epochs")
    parser.add_argument('--hyperparamter', type=int, default=10, help="loss of lambda")
    parser.add_argument('--lr', type=float, default=1e-4, help="learning rate")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device: ', device)
    
    trans = transforms.Compose([
        transforms.RandomCrop((256)),
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=1),
        transforms.Normalize((0.5,), (0.5))
    ])
    
    dataset = utils.Customdataset(transform=trans, vis_dataset=args.vis_dataset, ir_dataset=args.ir_dataset)
    train_dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    print('===> Loading Dataset Completed')
    
    fusion_model = nn.DataParallel(DenseFuse(), device_ids=[0, 1]) # 2 GPUs
    fusion_model.to(device) 
    optimizer = torch.optim.Adam(fusion_model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-8)
    
    mse_loss = nn.MSELoss().cuda()
    
    model_param_path = './saved_models'
    if not os.path.exists(model_param_path):
        os.mkdir(model_param_path)
    
    prev_time = time.time()
    
    for epoch in range(args.epochs):
        fusion_model.train()
        
        for batch, (vis_img, ir_img) in enumerate(train_dataloader):
            vis_img = vis_img.to(device)
            ir_img = ir_img.to(device)
            
            fusion_output = fusion_model(vis_img, ir_img).to(device)
            
            L_p = mse_loss(fusion_output, vis_img) + mse_loss(fusion_output, ir_img)
            L_ssim = (ssim(fusion_output, vis_img) + ssim(fusion_output, ir_img)) * 0.5
            
            total_loss = L_p + args.hyperparamter * (1-L_ssim)
            
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

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
                    L_ssim.item(),
                    total_loss.item(),
                    time_left,
                )
            )
        
        torch.save(fusion_model.state_dict(), "./saved_models/%s/model_fusion%d.pth" % ("basic", epoch))
    
if __name__ == "__main__":
    main()
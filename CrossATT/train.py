import argparse
import torch
import utils
import torch.nn as nn
import sys
import datetime
import time
import os
import torch.nn.functional as F
import losses

from model import Generator
from torch.utils.data import DataLoader
from torchvision import transforms
from pytorch_msssim import ssim # SSIM
from kornia.filters import SpatialGradient

def edge_detect(img):
    spatial = SpatialGradient('diff')
    edge_img = spatial(img)
    
    dx_fu, dy_fu = edge_img[:, :, 0, :, :], edge_img[:, :, 1, :, :]
    edge_img_output = torch.sqrt(torch.pow(dx_fu, 2) + torch.pow(dy_fu, 2))
    
    return edge_img_output

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--vis_dataset', type=str, default="Dataset/visible_20000", help="KAIST Dataset")
    parser.add_argument('--ir_dataset', type=str, default="Dataset/lwir_20000", help="KAIST Dataset")
    parser.add_argument('--batch_size', type=int, default=4, help="batch size for training")
    parser.add_argument('--epochs', type=int, default=20, help="number of training epochs")
    parser.add_argument('--hyperparamter', type=int, default=10, help="loss of lambda")
    parser.add_argument('--lr', type=float, default=1.5e-4, help="learning rate")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device: ', device)
    
    trans = transforms.Compose([
        transforms.RandomCrop((224)),
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=1),
        transforms.Normalize((0.5,), (0.5))
    ])
    
    dataset = utils.Customdataset(transform=trans, vis_dataset=args.vis_dataset, ir_dataset=args.ir_dataset)
    train_dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    print('===> Loading Dataset Completed')
    
    fusion_model = nn.DataParallel(Generator(), device_ids=[0]) # 2 GPUs
    fusion_model.to(device) 
    optimizer = torch.optim.Adam(fusion_model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-8)
    
    model_param_path = "/media/hdd/sungmin/Model/CrossATT/saved_models/Cmtfusion_loss"
    if not os.path.exists(model_param_path):
        os.mkdir(model_param_path)
    
    mse = nn.MSELoss()
    loss_p = losses.perceptual_loss().cuda()
    loss_spa = losses.L_spa().cuda()
    loss_fre = losses.frequency().cuda()
    
    prev_time = time.time()
    H, W = 224, 224

    for epoch in range(args.epochs):
        fusion_model.train()
        
        for batch, (vis_img, ir_img) in enumerate(train_dataloader):
            vis_img = vis_img.to(device)
            ir_img = ir_img.to(device)
            
            fusion_output = fusion_model(vis_img, ir_img).to(device)
            
            # edge_fusion_img = edge_detect(fusion_output)
            # edge_vis_img = edge_detect(vis_img)
            # edge_ir_img = edge_detect(ir_img)
            
            mse_loss = mse(fusion_output, vis_img) + mse(fusion_output, ir_img)
            fre_loss = loss_fre(fusion_output, vis_img.cuda(), ir_img.cuda())
            spa_loss = 0.5 * torch.mean(loss_spa(fusion_output, vis_img)) + 0.5 * torch.mean(
                loss_spa(fusion_output, ir_img))
            loss_per = 0.5 * loss_p(fusion_output, vis_img.cuda()) + 0.5 * loss_p(fusion_output, ir_img.cuda())
            fuse_loss = (mse_loss / 2) + 0.8 * spa_loss + 0.02 * loss_per + 0.05 * fre_loss
            total_loss = fuse_loss
            
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            batches_done = epoch * len(train_dataloader) + batch
            batches_left = args.epochs * len(train_dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()
            
            # 로그 출력
            sys.stdout.write(
                "\rTrain : [Epoch %d/%d] [Batch %d/%d] [fuse_loss: %f] [mse_loss: %f] [spa_loss: %f] [per_loss: %f] [total_loss: %f] ETA: %s"
                % (
                    epoch,
                    args.epochs,
                    batch,
                    len(train_dataloader),
                    fuse_loss.item(),
                    mse_loss.item(),
                    spa_loss.item(),
                    loss_per.item(),
                    total_loss.item(),
                    time_left,
                )
            )
        
        save_path = os.path.join(model_param_path, "model_fusion%d.pth" % epoch)
        torch.save(fusion_model.state_dict(), save_path)
        
if __name__ == "__main__":
    main()

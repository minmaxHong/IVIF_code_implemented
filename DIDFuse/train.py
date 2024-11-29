import argparse
import torch
import utils
import torch.nn as nn
import sys
import datetime
import time
import os
import kornia

from model import Encoder, Decoder
from torch.utils.data import DataLoader
from torchvision import transforms
from pytorch_msssim import ssim # SSIM

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vis_dataset', type=str, default=r"C:\Users\USER\Desktop\Dataset\KAIST", help="KAIST Dataset")
    parser.add_argument('--ir_dataset', type=str, default=r"C:\Users\USER\Desktop\Dataset\KAIST", help="KAIST Dataset")
    parser.add_argument('--batch_size', type=int, default=24, help="batch size for training")
    parser.add_argument('--epochs', type=int, default=120, help="number of training epochs")
    parser.add_argument('--a1', type=int, default=0.05, help="L1 a1 hyperparameter")
    parser.add_argument('--a2', type=int, default=2, help="L2 a2 hyperparameter")
    parser.add_argument('--a3', type=int, default=2, help="L2 a3 hyperparameter")
    parser.add_argument('--a4', type=int, default=10, help="L2 a4 hyperparameter")
    parser.add_argument('--lambd', type=int, default=, help="SSIM lambda")
    parser.add_argument('--lr', type=float, default=1e-1, help="learning rate")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device: ', device)
    
    trans = transforms.Compose([
        transforms.CenterCrop((128)),
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=1),
        transforms.Normalize((0.5, ), (0.5, ))
    ])
    dataset = utils.Customdataset(transform=trans, vis_dataset=args.vis_dataset, ir_dataset=args.ir_dataset)
    train_dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    print('===> Loading Dataset Completed')
    
    encoder_model = nn.DataParallel(Encoder(), device_ids=[0, 1])
    decoder_model = nn.DataParallel(Decoder(), device_ids=[0, 1])
    
    encoder_model.to(device)
    decoder_model.to(device)
    
    optim_encoder = torch.optim.Adam(encoder_model.parameters(), lr=args.lr)
    optim_decoder = torch.optim.Adam(decoder_model.parameters(), lr=args.lr)
    
    optim_encoder_scheduler = torch.optim.lr_scheduler.MultiStepLR(optim_encoder, [args.epochs // 3, (args.epochs // 3) * 2], gamma=0.1)
    optim_decoder_scheduler = torch.optim.lr_scheduler.MultiStepLR(optim_decoder, [args.epochs // 3, (args.epochs // 3) * 2], gamma=0.1)

    L1_term_loss = nn.L1Loss()
    L1_term_base_part_mse = nn.MSELoss()
    L1_term_detail_content_mse = nn.MSELoss()
    L2_term_vis_mse = nn.MSELoss()
    L2_term_ir_mse = nn.MSELoss()
    
    model_param_path = './saved_models'
    if not os.path.exists(model_param_path):
        os.mkdir(model_param_path)
    
    total_losses = []
    prev_time = time.time()
    for epoch in range(args.epochs):
        encoder_model.train()
        decoder_model.train()
        
        for batch, (vis_img, ir_img) in enumerate(train_dataloader):
            vis_img = vis_img.to(device)
            ir_img = ir_img.to(device)
            
            vis_base_part_fm, vis_detail_content_fm, vis_conv1_fm, vis_conv2_fm = encoder_model(vis_img) # Visible
            ir_base_part_fm, ir_detail_content_fm, ir_conv1_fm, ir_conv2_fm = encoder_model(ir_img) # Infrared
            
            vis_decoder_input = torch.cat((vis_base_part_fm, vis_detail_content_fm), dim=1)
            ir_decoder_input = torch.cat((ir_base_part_fm, ir_detail_content_fm), dim=1)
            
            vis_decoder_output = decoder_model(vis_decoder_input, vis_conv1_fm, vis_conv2_fm) 
            ir_decoder_output = decoder_model(ir_decoder_input, ir_conv1_fm, ir_conv2_fm) 
            
            L1 = (L1_term_base_part_mse(vis_base_part_fm, ir_base_part_fm)) - args.a1 * (L1_term_detail_content_mse(vis_detail_content_fm, ir_detail_content_fm))
            
            # L2-term loss
            vis_ssim = (1-ssim(vis_img, vis_decoder_output)) * 0.5 
            ir_ssim = (1-ssim(ir_img, ir_decoder_output)) * 0.5 
            
            vis_f = L2_term_vis_mse(vis_img, vis_decoder_output) + args.lambd * vis_ssim
            ir_f = L2_term_ir_mse(ir_img, ir_decoder_output) + args.lambd * ir_ssim
            
            gradient_loss = L1_term_loss(
                kornia.filters.SpatialGradient()(vis_img),
                kornia.filters.SpatialGradient()(vis_decoder_output)
            )
            
            L2 = args.a2 * ir_f + args.a3 * vis_f + args.a4 * gradient_loss
            
            total_loss = L1 + L2

            total_loss.backward()
            total_losses.append(total_loss)
            optim_encoder.step()
            optim_decoder.step()
            
            optim_encoder.zero_grad()
            optim_decoder.zero_grad()
            
            
            batches_done = epoch * len(train_dataloader) + batch
            batches_left = args.epochs * len(train_dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()
            
            sys.stdout.write(
                "\rTrain : [Epoch %d/%d] [Batch %d/%d] [L_1: %f] [L_2: %f] [vis_f: %f], [ir_f: %f] [vis_ssim: %f] [ir_ssim: %f] [Gradient_Loss: %f] [Total Loss: %f] ETA: %s"
                % (
                    epoch+1,
                    args.epochs,
                    batch,
                    len(train_dataloader)+1,
                    L1.item(),
                    L2.item(),
                    vis_f.item(),
                    ir_f.item(),
                    vis_ssim.item(),
                    ir_ssim.item(),
                    gradient_loss.item(),
                    total_loss.item(),
                    time_left,
                )
            )
        
        optim_encoder_scheduler.step()
        optim_decoder_scheduler.step()
        
        torch.save(encoder_model.state_dict(), "./saved_models/%s/VIS_AE%d.pth" % ("encoder", epoch+1))
        torch.save(decoder_model.state_dict(), "./saved_models/%s/IR_AE%d.pth" % ("decoder", epoch+1))


if __name__ == "__main__":
    main()    
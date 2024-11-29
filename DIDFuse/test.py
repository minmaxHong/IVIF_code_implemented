import os
from model import Encoder, Decoder, FusionLayer
import torch
from torchvision.utils import save_image
import utils
import argparse


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_path', type=str, default='./fusion_outputs/', help='path of fused image')
    parser.add_argument('--test_images', type=str, default=r'C:\Users\USER\Desktop\Dataset\TNO', help='path of source image')
    parser.add_argument('--dataset_name', type=str, default='basic', help='dataset name')
    parser.add_argument('--encoder_weight', type=str, default= r'C:\Users\USER\Desktop\code\sungmin_github\imagefusion_DIDFuse\encoder.pth', help='Encoder weights')
    parser.add_argument('--decoder_weight', type=str, default= r'C:\Users\USER\Desktop\code\sungmin_github\imagefusion_DIDFuse\decoder.pth', help='Decoder weights')
    args = parser.parse_args()

    if os.path.exists(args.out_path) is False:
        os.mkdir(args.out_path)

    # device setting for gpu users
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: ", device)

    encoder_model = torch.nn.DataParallel(Encoder(), device_ids=[0])
    decoder_model = torch.nn.DataParallel(Decoder(), device_ids=[0])
    encoder_model.load_state_dict(
        torch.load(args.encoder_weight, map_location=device), strict=False)
    decoder_model.load_state_dict(
        torch.load(args.decoder_weight, map_location=device), strict=False)
    
    fusion_layer = FusionLayer()
    
    print("===>Testing using weights:")
    
    encoder_model.cuda()
    decoder_model.cuda()
    
    encoder_model.eval()
    decoder_model.eval()

    with torch.no_grad():
        for i in range(len(args.test_images)):
            index = i + 1
            infrared_path = args.test_images + '/IR' + str(index) + '.png'
            visible_path = args.test_images + '/VIS' + str(index) + '.png'
            if os.path.isfile(infrared_path):
                real_ir_imgs = utils.get_test_images(infrared_path, height=None, width=None)
                real_rgb_imgs = utils.get_test_images(visible_path, height=None, width=None)

                torch.cuda.ipc_collect()
                torch.cuda.empty_cache()
                B_vis, D_vis, conv1_fm_vis, conv2_fm_vis = encoder_model(real_rgb_imgs.cuda())
                B_ir, D_ir, conv1_fm_ir, conv2_fm_ir = encoder_model(real_ir_imgs.cuda())
                
                B_ = fusion_layer(B_vis, B_ir)
                D_ = fusion_layer(D_vis, D_ir)
                conv1_fm_ = fusion_layer(conv1_fm_vis, conv1_fm_ir)
                conv2_fm_ = fusion_layer(conv2_fm_vis, conv2_fm_ir)
                
                fusion_output = decoder_model(torch.cat((B_, D_), dim=1), conv1_fm_, conv2_fm_)
                
                
                # # save images
                save_image(fusion_output, "fusion_outputs/%d.png" % index, normalize=True)

    print('Done......')


if __name__ == '__main__':
    run()
import os
import sys
from model import Generator
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
import utils
import argparse


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_path', type=str, default='./fusion_outputs/', help='path of fused image')
    parser.add_argument('--test_images', type=str, default='/media/hdd/sungmin/Test/TNO', help='path of source image')
    parser.add_argument('--dataset_name', type=str, default='basic', help='dataset name')
    parser.add_argument('--weights', type=str, default='/media/hdd/sungmin/Model/CrossATT/saved_models/Cmtfusion_loss/model_fusion1.pth', help='dataset name')
    args = parser.parse_args()

    if os.path.exists(args.out_path) is False:
        os.mkdir(args.out_path)

    # device setting for gpu users
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: ", device)

    fusion_model = torch.nn.DataParallel(Generator(), device_ids=[0])
    fusion_model.load_state_dict(
        torch.load(args.weights, map_location=device))
    print("===>Testing using weights: ", args.weights)
    fusion_model.cuda()
    fusion_model.eval()

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
                fused = fusion_model(real_rgb_imgs.cuda(), real_ir_imgs.cuda())
                print(f"index[{index}] was processed")
                # # save images
                output_tensor = F.interpolate(fused, size=(270, 360), mode='bilinear', align_corners=False)
                save_image(output_tensor, "fusion_outputs/%d.png" % index, normalize=True)

    print('Done......')

if __name__ == '__main__':
    run()
    
    2, 1.48
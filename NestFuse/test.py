import os
import torch
import torch.nn as nn
import argparse
import utils

from torchvision.utils import save_image
from model import NestFuse
from fusion_strategy import fusion

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_images", type=str, default=r'C:\Users\USER\Desktop\Dataset\TNO', help="path of source image")
    parser.add_argument("--out_path", type=str, default='./fusion_outputs', help='path of fused image')
    parser.add_argument("--weights", type=str, default=r"C:\Users\USER\Desktop\code\sungmin_github\imagefusion_NetFuse\saved_models\basic\model_fusion0.pth", help="NestFuse weights")
    args = parser.parse_args()
    
    # if os.path.exists(args.out_path):
        # os.mkdir(args.out_path)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fusion_model = NestFuse()
    checkpoint = torch.load(args.weights, map_location=device)
    
    new_state_dict = {}
    for key, value in checkpoint.items():
        new_key = key.replace('module.', '')  
        new_state_dict[new_key] = value
    
    fusion_model.load_state_dict(new_state_dict)
    fusion_model.to(device)
    fusion_model.eval()
    
    with torch.no_grad():
        for i in range(len(args.test_images)):
            index = i + 1
            infrared_path = args.test_images + '/IR' + str(index) + '.png'
            visible_path = args.test_images + '/VIS' + str(index) + '.png'
            
            if os.path.isfile(visible_path) and os.path.isfile(infrared_path):
                ir_img = utils.get_test_images(infrared_path)
                vis_img = utils.get_test_images(visible_path)
                ir_img = ir_img.to(device)
                vis_img = vis_img.to(device)
                
                
                _ECB10_vis, _ECB20_vis, _ECB30_vis, _ECB40_vis = fusion_model.encoder_ouputs(vis_img)
                _ECB10_ir, _ECB20_ir, _ECB30_ir, _ECB40_ir = fusion_model.encoder_ouputs(ir_img)
                
                _ECB10_fused = fusion(_ECB10_vis, _ECB10_ir)
                _ECB20_fused = fusion(_ECB20_vis, _ECB20_ir)
                _ECB30_fused = fusion(_ECB30_vis, _ECB30_ir)
                _ECB40_fused = fusion(_ECB40_vis, _ECB40_ir)
                
                fusion_img = fusion_model.decoder_outputs(_ECB10_fused, _ECB20_fused, _ECB30_fused, _ECB40_fused, is_eval=True)
                save_image(fusion_img, "fusion_outputs/%d.png" % index, normalize=True)
                
                print(f"{index},, Saving Fusion Images, {fusion_img.size()}")
                
    print("...Done")
    
if __name__ == "__main__":
    main()
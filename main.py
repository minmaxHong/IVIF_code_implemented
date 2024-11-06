import os
import torch
import cv2
import numpy as np

from base_part import BasePart
from detail_content import DetailContent

def load_image(visible_path=None, infrared_path=None):
    visible_image = cv2.imread(visible_path)
    infrared_image = cv2.imread(infrared_path)
    
    visible_image = cv2.cvtColor(visible_image, cv2.COLOR_BGR2GRAY)
    infrared_image = cv2.cvtColor(infrared_image, cv2.COLOR_BGR2GRAY)
    
    
    return visible_image, infrared_image
def main():
    CURR_DIR = os.getcwd()
    
    visible_path = os.path.join(CURR_DIR, 'VIS1.png')
    infrared_path = os.path.join(CURR_DIR, 'IR1.png')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*50)
    print(f"visible_path: {visible_path}\ninfrared_path: {infrared_path}")
    # print(f"optimized_visible_path: {optimized_visible_path}\noptimized_infrared_path: {optimized_infrared_path}")
    print("="*50)
    
    visible_image, infrared_image = load_image(visible_path, infrared_path)
    
    # base part
    base_part = BasePart(visible_image, infrared_image, device)
    optimized_visible_I_k_b, optimized_infrared_I_k_b = base_part.get_optimization_I_k_b(hyperparameter=5,iterations_per_epoch=30000)
    base_part_result = base_part.get_fusion_base_parts()
    
    # detail content part
    detail_content_part = DetailContent(visible_image, infrared_image, device=device)
    detail_content_part.get_visible_infrared_I_k_d(optimized_visible_I_k_b, optimized_infrared_I_k_b)
    
    detail_content_result = detail_content_part.get_fused_detail_content()
    
    fusion_result = base_part_result + detail_content_result
    fusion_result = fusion_result.astype(np.uint8)
    
    cv2.imwrite('fusion_result.png', fusion_result)
    cv2.imshow('fusion_result', fusion_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()
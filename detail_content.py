import torch
import numpy as np
import cv2
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F

from PIL import Image

def preprocess_image(image: np.ndarray):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.fromarray(image).convert('RGB') # 1 channel -> 3channel transform, type: numpy->PIL.Image
    image = transform(image).unsqueeze(0)
    return image    

class VGGFeatureExtractor(nn.Module):
    def __init__(self):
        super(VGGFeatureExtractor, self).__init__()
        
        self.layer_indices = [0, 5, 10, 19] # relu_1_1, relu_2_1, relu_3_1, relu_4_1
        self.vgg19 = models.vgg19(pretrained=True).features
    
    def forward(self, x):
        features = []
        for i, layer in enumerate(self.vgg19):
            x = layer(x)
            if i in self.layer_indices:
                features.append(x)
        return features
    
class DetailContent:
    def __init__(self, visibie_image: np.ndarray, infrared_image: np.ndarray, device: torch.device):
        self.visible_image = visibie_image
        self.infrared_image = infrared_image
        
        self.feature_extractor = VGGFeatureExtractor()
        self.feature_extractor.eval()
        
        self.visible_I_k_d = None
        self.infrared_I_k_d = None
    
    def get_visible_infrared_I_k_d(self, optimized_visible_I_k_b: np.ndarray, optimized_infrared_I_k_b: np.ndarray):
        ''' I_k_d = I - I_k_b
        Args:
            optimized_visible_I_k_b: visible I_k_b
            optimized_infrared_I_k_b: infrared I_k_b
        
        Returns:
            visible_I_k_d, infrared_I_k_d: I_k_d = I - I_k_b
        '''
        self.visible_I_k_d = self.visible_image - optimized_visible_I_k_b
        self.infrared_I_k_d = self.infrared_image - optimized_infrared_I_k_b
    
    def get_initial_acitivity_level_map(self):
        ''' first of all, calculate φ_k^{i,m} = φ_i(I_k^d)
        and calculate C_k^i(x,y) = ||φ_k^{i,1:M}(x,y)||_1
        '''
        # calculate φ_k^{i,m}, i∈{1,2}
        preprocess_visible_I_k_d = preprocess_image(self.visible_I_k_d)
        preprocess_infrared_I_k_d = preprocess_image(self.infrared_I_k_d)
        with torch.no_grad():
            visible_feature_maps = self.feature_extractor(preprocess_visible_I_k_d) # i∈{1,2,3,4}
            infrared_feature_maps = self.feature_extractor(preprocess_infrared_I_k_d) # i∈{1,2,3,4}
        
        visible_feature_i = {}
        visible_feature_i['i_1'] = visible_feature_maps[0].squeeze(0).cpu()  # torch.Tensor -> np.array transform
        visible_feature_i['i_2'] = visible_feature_maps[1].squeeze(0).cpu()
        visible_feature_i['i_3'] = visible_feature_maps[2].squeeze(0).cpu()
        visible_feature_i['i_4'] = visible_feature_maps[3].squeeze(0).cpu()

        infrared_feature_i = {}
        infrared_feature_i['i_1'] = infrared_feature_maps[0].squeeze(0).cpu()
        infrared_feature_i['i_2'] = infrared_feature_maps[1].squeeze(0).cpu()
        infrared_feature_i['i_3'] = infrared_feature_maps[2].squeeze(0).cpu()
        infrared_feature_i['i_4'] = infrared_feature_maps[3].squeeze(0).cpu()

        i_1_channel, i_1_height, i_1_width = infrared_feature_i['i_1'].shape
        i_2_channel, i_2_height, i_2_width = infrared_feature_i['i_2'].shape
        i_3_channel, i_3_height, i_3_width = infrared_feature_i['i_3'].shape
        i_4_channel, i_4_height, i_4_width = infrared_feature_i['i_4'].shape

        initial_visible_activity_map_i = {}
        initial_visible_activity_map_i['i_1'] = torch.zeros((i_1_height, i_1_width), dtype=torch.float32)
        initial_visible_activity_map_i['i_2'] = torch.zeros((i_2_height, i_2_width), dtype=torch.float32)
        initial_visible_activity_map_i['i_3'] = torch.zeros((i_3_height, i_3_width), dtype=torch.float32)
        initial_visible_activity_map_i['i_4'] = torch.zeros((i_4_height, i_4_width), dtype=torch.float32)

        initial_infrared_activity_map_i = {}
        initial_infrared_activity_map_i['i_1'] = torch.zeros((i_1_height, i_1_width), dtype=torch.float32)
        initial_infrared_activity_map_i['i_2'] = torch.zeros((i_2_height, i_2_width), dtype=torch.float32)
        initial_infrared_activity_map_i['i_3'] = torch.zeros((i_3_height, i_3_width), dtype=torch.float32)
        initial_infrared_activity_map_i['i_4'] = torch.zeros((i_4_height, i_4_width), dtype=torch.float32)
        
        # visible, infrared- C^i_k(x,y)
        for key in visible_feature_i:
            visible_summed_l1_norm = torch.norm(visible_feature_i[key], p=1, dim=0)
            infrared_summed_l1_norm = torch.norm(infrared_feature_i[key], p=1, dim=0)
            
            height, width = visible_summed_l1_norm.shape
            for i in range(height):
                for j in range(width):
                    initial_visible_activity_map_i[key][i][j] = visible_summed_l1_norm[i][j]
                    initial_infrared_activity_map_i[key][i][j] = infrared_summed_l1_norm[i][j]
        
        return initial_visible_activity_map_i, initial_infrared_activity_map_i
    
    def get_final_acitivity_level_map(self, block_size: int=1):
        ''' C^{hat}^i_k(x,y) = C^i_k(x+b, y+θ) / (2r+1)^2, r: block_size
        '''
        initial_visible_activity_map_i, initial_infrared_activity_map_i = self.get_initial_acitivity_level_map()
        
        final_visible_activity_level_map_i = {}
        final_infrared_activity_level_map_i = {}
        
        # calculate final weight map -> visible, infrared 
        for key in initial_visible_activity_map_i:
            visible_C_k_i = initial_visible_activity_map_i[key]
            infrared_C_k_i = initial_infrared_activity_map_i[key]
            
            height, width = visible_C_k_i.shape
            
            final_visible_activity_level_map_i[key] = torch.zeros((height, width), dtype=torch.float32)
            final_infrared_activity_level_map_i[key] = torch.zeros((height, width), dtype=torch.float32)
            
            for i in range(1, height-1):
                for j in range(1, width-1):
                    visible_conv_sum = 0
                    infrared_conv_sum = 0
                    for beta in range(-block_size, block_size+1):
                        for theta in range(-block_size, block_size+1):
                            visible_conv_sum += visible_C_k_i[i+beta][j+theta]
                            infrared_conv_sum += infrared_C_k_i[i+beta][j+theta]
                    # print("*"*20)
                    # print(f'visible_conv_sum: , {visible_conv_sum},infrared_conv_sum: {infrared_conv_sum}')
                    # print(f'vis_activity_map: {visible_conv_sum / ((2*block_size + 1) ** 2)}, infrared_activity_map:{infrared_conv_sum / ((2*block_size + 1) ** 2)}' )
                    # print("*"*20)
                    
                    final_visible_activity_level_map_i[key][i][j] = visible_conv_sum / ((2*block_size + 1) ** 2)
                    final_infrared_activity_level_map_i[key][i][j] = infrared_conv_sum / ((2*block_size + 1) ** 2)
        
        return final_visible_activity_level_map_i, final_infrared_activity_level_map_i
        
    def get_initial_weight_map(self, eps: int=1e-05):
        '''get W^i_k(x,y) = C_{hat}^i_k(x,y) / sigma C_{hat}^i_n(x,y)
        '''
        final_visible_activity_level_map_i, final_infrared_activity_level_map_i = self.get_final_acitivity_level_map()
        
        initial_visible_weight_map = {}
        initial_infrared_weight_map = {}
        
        for key in final_visible_activity_level_map_i.keys():
            height, width = final_visible_activity_level_map_i[key].shape
            
            initial_visible_weight_map[key] = torch.zeros((height, width), dtype=torch.float32)
            initial_infrared_weight_map[key] = torch.zeros((height, width), dtype=torch.float32)
            
            for i in range(height):
                for j in range(width):
                    sum = final_visible_activity_level_map_i[key][i][j] + final_infrared_activity_level_map_i[key][i][j]
                    
                    initial_visible_weight_map[key][i][j] = final_visible_activity_level_map_i[key][i][j] / (sum+eps)
                    initial_infrared_weight_map[key][i][j] = final_infrared_activity_level_map_i[key][i][j] / (sum+eps)
        
        return initial_visible_weight_map, initial_infrared_weight_map
    
    def get_final_weight_map(self):
        ''' input detail size -> height: 270, width: 360
        calculate upsampling
        '''
        initial_visible_weight_map, initial_infrared_weight_map = self.get_initial_weight_map()
        height, width = 270, 360
        final_visible_weight_map = {}
        final_infrared_weight_map = {}
        
        for key in initial_visible_weight_map.keys():
            before_interpolate_final_visible_weight_map = initial_visible_weight_map[key]
            before_interpolate_final_visible_weight_map = before_interpolate_final_visible_weight_map.unsqueeze(0).unsqueeze(0)
            
            before_interpolate_final_infrared_weight_map = initial_infrared_weight_map[key]
            before_interpolate_final_infrared_weight_map = before_interpolate_final_infrared_weight_map.unsqueeze(0).unsqueeze(0)
            
            after_interpolate_final_visible_weight_map = F.interpolate(before_interpolate_final_visible_weight_map, size=(height,width), mode='nearest')
            after_interpolate_final_infrared_weight_map = F.interpolate(before_interpolate_final_infrared_weight_map, size=(height,width), mode='nearest')
            
            final_visible_weight_map[key] = after_interpolate_final_visible_weight_map
            final_infrared_weight_map[key] = after_interpolate_final_infrared_weight_map
        
        return final_visible_weight_map, final_infrared_weight_map

    def get_fused_detail_content(self):
        ''' F^i_d(x,y) = sigma W^i_n(x,y) * I^d_n(x,y) -> F_d(x,y) = max[F^i_d(x,y)|i∈{1,2,3,4}]
        '''
        final_visible_weight_map, final_infrared_weight_map = self.get_final_weight_map()
        
        # visible-I^d_n(x,y): self.visible_I_k_d
        # infrared-I^d_n(x,y): self.infrared_I_k_d
        fused_detail_content_i = {}
        for key in final_visible_weight_map.keys():
            visible_sum = final_visible_weight_map[key] * self.visible_I_k_d
            infrared_sum = final_infrared_weight_map[key] * self.infrared_I_k_d
            
            fused_detail_content_i[key] = visible_sum + infrared_sum
        
        fused_detail_content = torch.max(torch.stack(list(fused_detail_content_i.values())), dim=0).values # F_d(x,y) = max[F^i_d(x,y)|i∈{1,2,3,4}]
        fused_detail_content = fused_detail_content.squeeze(0).squeeze(0)

        np_fused_detail_content = fused_detail_content.numpy()
        np_fused_detail_content = np_fused_detail_content.astype(np.uint8)
        # cv2.imwrite('detail_content.png', np_fused_detail_content)
        # cv2.imshow('image', np_fused_detail_content)
        # cv2.waitKey(0)

        return np_fused_detail_content
                   
def load_image(visible_path=None, infrared_path=None):
    visible_image = cv2.imread(visible_path)
    infrared_image = cv2.imread(infrared_path)
    
    visible_image = cv2.cvtColor(visible_image, cv2.COLOR_BGR2GRAY)
    infrared_image = cv2.cvtColor(infrared_image, cv2.COLOR_BGR2GRAY)
    
    return visible_image, infrared_image
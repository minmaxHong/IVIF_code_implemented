import torch
import torch.nn as nn
import torch.nn.functional as F

from torchinfo import summary

class Encoder(nn.Module):
    def __init__(self, out_channels=16, nb_filters=[1, 16, 32, 48]):
        super(Encoder, self).__init__()
        
        out_channels = 16
        nb_filters = nb_filters
        
        self.Pad = nn.ReflectionPad2d(1)
        self.vis_C1_conv = nn.Conv2d(in_channels=nb_filters[0], out_channels=out_channels, kernel_size=3, stride=1, padding=0)
        self.vis_DC1_conv = nn.Conv2d(in_channels=nb_filters[1], out_channels=out_channels, kernel_size=3, stride=1, padding=0)
        self.vis_DC2_conv = nn.Conv2d(in_channels=nb_filters[2], out_channels=out_channels, kernel_size=3, stride=1, padding=0)
        self.vis_DC3_conv = nn.Conv2d(in_channels=nb_filters[3], out_channels=out_channels, kernel_size=3, stride=1, padding=0)
        
        self.ir_C1_conv = nn.Conv2d(in_channels=nb_filters[0], out_channels=out_channels, kernel_size=3, stride=1, padding=0)
        self.ir_DC1_conv = nn.Conv2d(in_channels=nb_filters[1], out_channels=out_channels, kernel_size=3, stride=1, padding=0)
        self.ir_DC2_conv = nn.Conv2d(in_channels=nb_filters[2], out_channels=out_channels, kernel_size=3, stride=1, padding=0)
        self.ir_DC3_conv = nn.Conv2d(in_channels=nb_filters[3], out_channels=out_channels, kernel_size=3, stride=1, padding=0)
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward_vis(self, vis_img: torch.Tensor):
        ''' forward/backprop for visible image
        Args:
            vis_img: visible image
        
        Return:
            fuesed_output: feature maps for After Convolution visible image
        '''
        vis_img = self.Pad(vis_img)
        C1 = self.relu(self.vis_C1_conv(vis_img))
        
        # Dense Block
        DC1_input = self.Pad(C1)
        DC1_output = self.relu(self.vis_DC1_conv(DC1_input))
        
        DC2_input = torch.cat([C1, DC1_output], dim=1)
        DC2_input = self.Pad(DC2_input)
        DC2_output = self.relu(self.vis_DC2_conv(DC2_input))
        
        DC3_input = torch.cat([C1, DC1_output, DC2_output], dim=1)
        DC3_input = self.Pad(DC3_input)
        DC3_output = self.relu(self.vis_DC3_conv(DC3_input))
        
        fused_output = torch.cat([C1, DC1_output, DC2_output, DC3_output], dim=1)     
        return fused_output
    
    def forward_ir(self, infrared_img: torch.Tensor):
        ''' forward/backprop for infrared image
        Args:
            vis_img: infrared image
        
        Return:
            fuesed_output: feature maps for After Convolution infrared image
        '''
        infrared_img = self.Pad(infrared_img)
        C1 = self.relu(self.ir_C1_conv(infrared_img))
        
        # Dense Block
        DC1_input = self.Pad(C1)
        DC1_output = self.relu(self.ir_DC1_conv(DC1_input))
        
        DC2_input = torch.cat([C1, DC1_output], dim=1)
        DC2_input = self.Pad(DC2_input)
        DC2_output = self.relu(self.ir_DC2_conv(DC2_input))
        
        DC3_input = torch.cat([C1, DC1_output, DC2_output], dim=1)
        DC3_input = self.Pad(DC3_input)
        DC3_output = self.relu(self.ir_DC3_conv(DC3_input))
        
        fused_output = torch.cat([C1, DC1_output, DC2_output, DC3_output], dim=1)     
        return fused_output
    
    def forward(self, vis_img: torch.Tensor, infrared_img: torch.Tensor) -> torch.Tensor:
        vis_feature_maps = self.forward_vis(vis_img)
        ir_feature_maps = self.forward_ir(infrared_img)

        return vis_feature_maps, ir_feature_maps


class AdditionStrategyFusionLayer(nn.Module):
    def __init__(self):
        super(AdditionStrategyFusionLayer, self).__init__()
        self.vis_feature_maps = None
        self.ir_feature_maps = None

    def init_feature_maps(self, vis_feature_maps: torch.Tensor, ir_feature_maps: torch.Tensor):
        ''' initialized each feature maps of source images
        Args:
            vis_feature_maps: feature map of visible
            ir_feature_maps: feature map of ir
        '''
        self.vis_feature_maps = vis_feature_maps
        self.ir_feature_maps = ir_feature_maps
    
    def get_featuremaps(self):
        ''' ==> Addition Strategy formula: f^m(x,y) = \sum_{i=1}^k φ_i^m(x,y)
        '''
        f_m = self.vis_feature_maps + self.ir_feature_maps
        return f_m

class L1NormFusionLayer(nn.Module):
    def __init__(self):
        super(L1NormFusionLayer, self).__init__()
        self.vis_feature_maps = None
        self.ir_feature_maps = None
        self.Pad = nn.ReflectionPad2d(1)
    
    def get_feature_maps(self, vis_feature_maps: torch.Tensor, ir_feature_maps: torch.Tensor):
        self.vis_feature_maps = vis_feature_maps
        self.ir_feature_maps = ir_feature_maps
    
    def get_initial_activity_level_map(self):
        ''' C_i(x,y) = ||φ_i^{1:M}(x,y)||_1
        '''
        vis_initial_activity_level_map = torch.sum(self.vis_feature_maps, dim=1)
        ir_initial_activity_level_map = torch.sum(self.ir_feature_maps, dim=1)

        return vis_initial_activity_level_map, ir_initial_activity_level_map

    def get_final_activity_level_map(self, vis_initial_activity_level_map: torch.Tensor, ir_initial_activity_level_map: torch.Tensor, kernel_size:int =1):
        '''block-based average operator
        Args:
            vis_initial_activity_level_map: C_i(x,y) for vis
            ir_initial_activity_level_map: C_i(x,y) for ir
        
        Returns:
            vis_final_activity_level_map: \{hat}C^i(x,y) for vis
            ir_final_activity_level_map: \{hat}C^i(x,y) for ir
        '''
        vis_initial_activity_level_map = vis_initial_activity_level_map
        ir_initial_activity_level_map = ir_initial_activity_level_map
        
        batch = vis_initial_activity_level_map.size(0)
        
        # block-based average operator
        kernel_size = (2*kernel_size+1, 2*kernel_size+1)
        kernel = torch.ones(batch, 1, *kernel_size).cuda()
        
        vis_padded = self.Pad(vis_initial_activity_level_map)
        ir_padded = self.Pad(ir_initial_activity_level_map)
        
        vis_summed = F.conv2d(vis_padded, kernel, stride=1, padding=0, groups=batch) # groups: Can independent calculation batch
        ir_summed = F.conv2d(ir_padded, kernel, stride=1, padding=0, groups=batch)
        
        normalized_val = kernel_size[0] * kernel_size[1]
        
        vis_final_activity_level_map = vis_summed / normalized_val
        ir_final_activity_level_map = ir_summed / normalized_val
        # ===============================
        
        return vis_final_activity_level_map, ir_final_activity_level_map    
    
    def get_fused_feature_maps(self, vis_final_activty_level_map: torch.Tensor, ir_final_activity_level_map: torch.Tensor):
        ''' soft-max operation, weighted sum with Decoder Input
        Args:
            vis_final_activity_level_map: \{hat}C^i(x,y) for vis
            ir_final_activity_level_map: \{hat}C^i(x,y) for ir
            
        Returns:
            f_m: final fused image

        '''
        vis_ir_weighted_sum = vis_final_activty_level_map + ir_final_activity_level_map
        
        w_vis = vis_final_activty_level_map / vis_ir_weighted_sum
        w_ir = ir_final_activity_level_map / vis_ir_weighted_sum
        
        w_vis = w_vis.unsqueeze(1)
        w_ir = w_ir.unsqueeze(1)
        
        vis_weight = w_vis * self.vis_feature_maps
        ir_weight = w_ir * self.ir_feature_maps
        
        f_m = vis_weight + ir_weight    
        return f_m

    def activate_fusion_layer(self):
        vis_initial_activity_level_map, ir_initial_activity_level_map = self.get_initial_activity_level_map()
        vis_final_activity_level_map, ir_final_activity_level_map = self.get_final_activity_level_map(vis_initial_activity_level_map, ir_initial_activity_level_map)
        
        return self.get_fused_feature_maps(vis_final_activity_level_map, ir_final_activity_level_map)
        
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        
        self.Pad = nn.ReflectionPad2d(1)
        self.C2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.C3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.C4 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=0)
        self.C5 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=0)
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x:torch.Tensor):
        x = self.Pad(x)
        C2 = self.relu(self.C2(x))
        
        C2 = self.Pad(C2)
        C3 = self.relu(self.C3(C2))
        
        C3 = self.Pad(C3)
        C4 = self.relu(self.C4(C3))
        
        C4 = self.Pad(C4)
        fusion_output = self.C5(C4)
        return fusion_output
        

class DenseFuse(nn.Module):
    def __init__(self):
        super(DenseFuse, self).__init__()
        
        self.encoder = Encoder()
        self.addtion_fusionlayer = AdditionStrategyFusionLayer()
        self.l1_norm_fusionlayer = L1NormFusionLayer()
        self.decoder = Decoder()
    
    def forward(self, vis_img: torch.Tensor, ir_img: torch.Tensor):
        ''' Pipeline: Encoder => FusionLayer => Decoder
        Args:
            vis_img: visible image
            ir_img: infrared image
        
        Returns:
            fusion_output: Also Decoder output and fusion image
        '''
        # Encoder
        # outputs: each feature maps
        vis_feature_maps, ir_feature_maps = self.encoder(vis_img, ir_img)
        
        # FusionLayer (addtion, l1_norm) => 2 version of fusion strategy
        # self.fusionlayer.init_feature_maps(vis_feature_maps, ir_feature_maps) # addition fusion strategy
        # combined_feature_maps = self.fusionlayer.get_featuremaps()
        self.l1_norm_fusionlayer.get_feature_maps(vis_feature_maps, ir_feature_maps) # l1_norm, soft-max strategy
        combined_feature_maps = self.l1_norm_fusionlayer.activate_fusion_layer()
        
        # Decoder
        # outputs: fusion image
        fusion_output = self.decoder(combined_feature_maps)
        # print(f"fusion output: ", fusion_output)
        return fusion_output

 
def main():
    # if you want check the model parameter, remove annotation
    model = DenseFuse()
    summary(model, input_size=[(1, 1, 256, 256), (1, 1, 256, 256)])

if __name__ == "__main__":
    main()
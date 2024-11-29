import torch
import torch.functional as F

def spatial_attention(vis_features: torch.Tensor, ir_features: torch.Tensor, epsilon=1e-05):
    shape = vis_features.size(1)

    vis_features_mean = vis_features.mean(dim=1, keepdim=True)
    ir_features_mean = ir_features.mean(dim=1, keepdim=True)
    
    vis_weightingMaps = torch.exp(vis_features_mean) / (torch.exp(vis_features_mean) + torch.exp(ir_features_mean) + epsilon)
    ir_weightingMaps = torch.exp(ir_features_mean) / (torch.exp(vis_features_mean) + torch.exp(ir_features_mean) + epsilon)
    
    vis_weightingMaps = vis_weightingMaps.repeat(1, shape, 1, 1) # 같은 값 channel마다 생성
    ir_weightingMaps = ir_weightingMaps.repeat(1, shape, 1, 1)
    
    return vis_features * vis_weightingMaps + ir_features * ir_weightingMaps
    
def channel_attention(vis_features: torch.Tensor, ir_features: torch.Tensor, epsilon=1e-05):
    vis_gap = torch.mean(vis_features.view(vis_features.size(0), vis_features.size(1), -1), dim=2).unsqueeze(2).unsqueeze(2)
    ir_gap = torch.mean(ir_features.view(ir_features.size(0), ir_features.size(1), -1), dim=2).unsqueeze(2).unsqueeze(2)
    
    vis_weightingMaps = torch.exp(vis_gap) / (torch.exp(vis_gap) + torch.exp(ir_gap) + epsilon)
    ir_weightingMaps = torch.exp(ir_gap) / (torch.exp(vis_gap) + torch.exp(ir_gap) + epsilon)
    
    vis_weightingMaps = vis_weightingMaps.repeat(1, 1, vis_features.size(2), vis_features.size(3))
    ir_weightingMaps = ir_weightingMaps.repeat(1, 1, ir_features.size(2), ir_features.size(3))
    
    # print(f"vis_features: {vis_features.size()}, vis_weightingMaps: {vis_weightingMaps.size()}")
    
    return vis_features * vis_weightingMaps + ir_features * ir_weightingMaps
    

def fusion(vis_features: torch.Tensor, ir_features: torch.Tensor):
    spatial_value = spatial_attention(vis_features, ir_features)
    channel_value = channel_attention(vis_features, ir_features)
    
    # print(f"spatial_value: {spatial_value.size()}, channel_value: {channel_value.size()}")
    return (spatial_value + channel_value) * 0.5
    
    
def main():
    vis_features = torch.randn((1, 64, 128, 128))
    ir_features = torch.randn((1, 64, 128, 128))
    
    fusion(vis_features, ir_features)

if __name__ == "__main__":
    main()
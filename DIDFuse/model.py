import torch
import torch.nn as nn
import torch.nn.functional as F

from torchinfo import summary

# Conv1
class Conv1(nn.Module):
    def __init__(self, in_channels=1, out_channels=64):
        super(Conv1, self).__init__()
        self.layers = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, stride=1, kernel_size=3, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )
        
    def forward(self, x: torch.Tensor):
        return self.layers(x)

# Conv2, 5, 6
class Conv2_5_6(nn.Module):
    def __init__(self, in_channels=64, out_channels=128):
        super(Conv2_5_6, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )
    def forward(self, x: torch.Tensor):
        return self.layers(x)
    
# Conv3, 4
class Conv3_4(nn.Module):
    def __init__(self, in_channels=64, out_channels=64):
        super(Conv3_4, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Tanh()
        )
    def forward(self, x: torch.Tensor):
        return self.layers(x)

# Conv7
class Conv7(nn.Module):
    def __init__(self, in_channels=64, out_channels=1):
        super(Conv7, self).__init__()
        self.layers = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, stride=1, kernel_size=3, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid()
        )
    def forward(self, x: torch.Tensor):
        return self.layers(x)

class Encoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=64):
        super(Encoder, self).__init__()
        self.Conv1 = Conv1(in_channels=in_channels, out_channels=out_channels)
        self.Conv2 = Conv2_5_6(in_channels=out_channels, out_channels=out_channels)
        self.Conv3 = Conv3_4(in_channels=out_channels, out_channels=out_channels)
        self.Conv4 = Conv3_4(in_channels=out_channels, out_channels=out_channels)
        
    def forward(self, x: torch.Tensor):
        conv1_fm = self.Conv1(x) # concatenate with Conv5 feature map
        conv2_fm = self.Conv2(conv1_fm) # concatenate with Conv6 feature map
        
        base_part_fm = self.Conv3(conv2_fm) 
        detail_content_fm = self.Conv4(conv2_fm)
        
        return base_part_fm, detail_content_fm , conv1_fm, conv2_fm # base_part feature map / detail content feature map / conv5 connection / conv6 connection

class Decoder(nn.Module):
    def __init__(self, in_channels=128, out_channels=64):
        super(Decoder, self).__init__()
        self.Conv5 = Conv2_5_6(in_channels=in_channels, out_channels=out_channels)
        self.Conv6 = Conv2_5_6(in_channels=out_channels*2, out_channels=out_channels)
        self.Conv7 = Conv7(in_channels=out_channels*2, out_channels=1)
        
    def forward(self, x: torch.Tensor, conv1_fm: torch.Tensor, conv2_fm: torch.Tensor):
        conv5_fm = self.Conv5(x)
        conv5_fm = torch.cat((conv5_fm, conv2_fm), dim=1)
        
        conv6_fm = self.Conv6(conv5_fm)
        conv6_fm = torch.cat((conv6_fm, conv1_fm), dim=1)
        
        output_fm = self.Conv7(conv6_fm)
        return output_fm

class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.Encoder = Encoder()
        self.Decoder = Decoder()
    
    def forward(self, x: torch.Tensor):
        # Encoder
        en_base_part_fm, en_detail_content_fm, conv1_fm, conv2_fm = self.Encoder(x)
        
        # Decoder
        decoder_input = torch.cat((en_base_part_fm, en_detail_content_fm), dim=1)
        decoder_output = self.Decoder(decoder_input, conv1_fm, conv2_fm)
        
        return decoder_output        

class FusionLayer(nn.Module):
    def __init__(self):
        pass
    
    def forward(self, vis: torch.Tensor, ir: torch.Tensor):
        box_filter = torch.ones((1, 1, 3, 3)) / 9
        box_filter = box_filter.repeat(vis.size(), 1, 1, 1)
        
        box_vis = F.conv2d(vis, box_filter, padding=1, groups=vis.size(1))
        box_ir = F.conv2d(ir, box_filter, padding=1, groups=ir.size(1))
        
        sum_box = box_vis + box_ir
        weight_vis = box_vis / sum_box
        weight_ir = box_ir / sum_box
        
        return (weight_vis * vis) + (weight_ir * ir)
        

def main():
    MODEL = AE()
    summary(MODEL, input_size=(1, 1, 128, 128))
    
if __name__ == "__main__":
    main()
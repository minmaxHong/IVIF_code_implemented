import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from torchinfo import summary
from torchvision.utils import save_image

class UpsampleReshape_eval(torch.nn.Module):
    def __init__(self):
        super(UpsampleReshape_eval, self).__init__()
        self.up = nn.Upsample(scale_factor=2)

    def forward(self, x1, x2):
        x2 = self.up(x2)
        shape_x1 = x1.size()
        shape_x2 = x2.size()
        left = 0
        right = 0
        top = 0
        bot = 0
        if shape_x1[3] != shape_x2[3]:
            lef_right = shape_x1[3] - shape_x2[3]
            if lef_right%2 == 0.0:
                left = int(lef_right/2)
                right = int(lef_right/2)
            else:
                left = int(lef_right / 2)
                right = int(lef_right - left)

        if shape_x1[2] != shape_x2[2]:
            top_bot = shape_x1[2] - shape_x2[2]
            if top_bot%2 == 0.0:
                top = int(top_bot/2)
                bot = int(top_bot/2)
            else:
                top = int(top_bot / 2)
                bot = int(top_bot - top)

        reflection_padding = [left, right, top, bot]
        reflection_pad = nn.ReflectionPad2d(reflection_padding)
        x2 = reflection_pad(x2)
        return x2
    
# NestFuse Convolution Operation
class Convolution(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super(Convolution, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.dropout = nn.Dropout2d(p=0.5)
        self.is_last = is_last

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        if self.is_last is False:
            out = F.relu(out, inplace=True)
        return out

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ConvBlock, self).__init__()
        
        block = []
        block += [Convolution(in_channels, in_channels // 2, kernel_size, stride),
                 Convolution(in_channels // 2, out_channels, 1, stride)]
        
        self.layer = nn.Sequential(*block)
    
    def forward(self, x: torch.Tensor):
        return self.layer(x)

# Encoder
class Encoder(nn.Module):
    def __init__(self, kernel_size=3, stride=1):
        super(Encoder, self).__init__()
        nb_filters = [16, 64, 112, 160, 208]
        self.ECB10 = ConvBlock(nb_filters[0], nb_filters[1], kernel_size, stride)
        self.ECB20 = ConvBlock(nb_filters[1], nb_filters[2], kernel_size, stride)
        self.ECB30 = ConvBlock(nb_filters[2], nb_filters[3], kernel_size, stride)
        self.ECB40 = ConvBlock(nb_filters[3], nb_filters[4], kernel_size, stride)        
        
        self.pool = nn.MaxPool2d(2, 2)
        
    def forward(self, x: torch.Tensor):
        _ECB10 = self.ECB10(x)
        _ECB20 = self.pool(self.ECB20(_ECB10))
        _ECB30 = self.pool(self.ECB30(_ECB20))
        _ECB40 = self.pool(self.ECB40(_ECB30))
        
        return _ECB10, _ECB20, _ECB30, _ECB40

class NestFuse(nn.Module):
    def __init__(self, _1x1_conv=1, _3x3_kernel_size=3, stride=1):
        super(NestFuse, self).__init__()
        input_channel, out_channels = 1, 16 # first
        _input_channel, _out_channels = 64, 1 # second
        
        _1stage_filters = [
                        [176, 64], # DCB11
                        [240, 64], # DCB12
                        [304, 64]] # DCB13
        _2stage_filters = [
                        [272, 112], # DCB21
                        [384, 112], # DCB22
                            ]
        _3stage_filter = [[368, 160]] # DCB31
        
        self.input_conv = Convolution(input_channel, out_channels, _1x1_conv, stride)
        self.output_conv = Convolution(_input_channel, _out_channels, _1x1_conv, stride, is_last=True)
        
        # Encoder
        self.Encoder = Encoder()

        # Decoder
        self.DCB11 = ConvBlock(_1stage_filters[0][0], _1stage_filters[0][1], _3x3_kernel_size, stride)
        self.DCB12 = ConvBlock(_1stage_filters[1][0], _1stage_filters[1][1], _3x3_kernel_size, stride)
        self.DCB13 = ConvBlock(_1stage_filters[2][0], _1stage_filters[2][1], _3x3_kernel_size, stride)
        self.DCB21 = ConvBlock(_2stage_filters[0][0], _2stage_filters[0][1], _3x3_kernel_size, stride)
        self.DCB22 = ConvBlock(_2stage_filters[1][0], _2stage_filters[1][1], _3x3_kernel_size, stride)
        self.DCB31 = ConvBlock(_3stage_filter[0][0], _3stage_filter[0][1], _3x3_kernel_size, stride)
        
        # Upsampling Model
        self.upsampling = nn.Upsample(scale_factor=2) # nearest upsampling
        self.upsample_eval = UpsampleReshape_eval()
    
    def encoder_ouputs(self, x: torch.Tensor):
        _Conv_input = self.input_conv(x) 
        _ECB10, _ECB20, _ECB30, _ECB40 = self.Encoder(_Conv_input)
    
        return _ECB10, _ECB20, _ECB30, _ECB40

    def decoder_outputs(self, _ECB10, _ECB20, _ECB30, _ECB40, is_eval=False):
        if is_eval:
            _DCB31 = self.DCB31(torch.cat((_ECB30, self.upsample_eval(_ECB30, _ECB40)), dim=1))
            _DCB21 = self.DCB21(torch.cat((_ECB20, self.upsample_eval(_ECB20, _ECB30)), dim=1))
            _DCB22 = self.DCB22(torch.cat((_ECB20, _DCB21, self.upsample_eval(_DCB21, _DCB31)), dim=1))
            _DCB11 = self.DCB11(torch.cat((_ECB10, self.upsample_eval(_ECB10, _ECB20)), dim=1))
            _DCB12 = self.DCB12(torch.cat((_ECB10, _DCB11, self.upsample_eval(_DCB11, _DCB21)), dim=1))
            _DCB13 = self.DCB13(torch.cat((_ECB10, _DCB11, _DCB12, self.upsample_eval(_DCB12, _DCB22)), dim=1))
            
            # save_image(torch.mean(_DCB31, dim=1), "fusion_outputs/_DCB31.png", normalize=True)
            # save_image(torch.mean(_DCB21, dim=1), "fusion_outputs/_DCB21.png", normalize=True)
            # save_image(torch.mean(_DCB22, dim=1), "fusion_outputs/_DCB22.png", normalize=True)
            # save_image(torch.mean(_DCB11, dim=1), "fusion_outputs/_DCB11.png", normalize=True)
            # save_image(torch.mean(_DCB12, dim=1), "fusion_outputs/_DCB12.png", normalize=True)
            # save_image(torch.mean(_DCB13, dim=1), "fusion_outputs/_DCB13.png", normalize=True)
            
            result = self.output_conv(_DCB13)
            
            return result
            
        else:   
            _DCB31 = self.DCB31(torch.cat((_ECB30, self.upsampling(_ECB40)), dim=1))
            _DCB21 = self.DCB21(torch.cat((_ECB20, self.upsampling(_ECB30)), dim=1))
            _DCB22 = self.DCB22(torch.cat((_DCB21, _ECB20, self.upsampling(_DCB31)), dim=1))
            _DCB11 = self.DCB11(torch.cat((_ECB10, self.upsampling(_ECB20)), dim=1))
            _DCB12 = self.DCB12(torch.cat((_DCB11, _DCB11, self.upsampling(_DCB21)), dim=1))
            _DCB13 = self.DCB13(torch.cat((_ECB10, _DCB11, _DCB12, self.upsampling(_DCB22)), dim=1))
            
            result = self.output_conv(_DCB13)
           
            return result
    
    def forward(self, x: torch.Tensor):
        _ECB10, _ECB20, _ECB30, _ECB40 = self.encoder_ouputs(x)
        _fusion_output = self.decoder_outputs(_ECB10,_ECB20,_ECB30,_ECB40)
        return _fusion_output
        

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def main():
    model = NestFuse()
    summary(model, input_size=[(1, 1, 256, 256)])
    
if __name__ == "__main__":
    main()
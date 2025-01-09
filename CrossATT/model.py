import torch
import torch.nn as nn

from torchinfo import summary
from intra_encoder import ViT
from inter_encoder import CrossVit
from einops import rearrange

class Auxiliary_Encoder(nn.Module):
    def __init__(self):
        super(Auxiliary_Encoder, self).__init__()
        nb_filters = [2, 4, 4, 4, 4]
        self.conv_layer_1 = self._conv_layer(in_channels=nb_filters[0], out_channels=nb_filters[1])
        self.conv_layer_2 = self._conv_layer(in_channels=nb_filters[1], out_channels=nb_filters[2])
        self.conv_layer_3 = self._conv_layer(in_channels=nb_filters[1]+nb_filters[2], out_channels=nb_filters[3])
        self.conv_layer_4 = self._conv_layer(in_channels=nb_filters[1]+nb_filters[2]+nb_filters[3], out_channels=nb_filters[4])
        
        
    def _conv_layer(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        conv_layer = []
        conv_layer.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding))
        conv_layer.append(nn.BatchNorm2d(out_channels))
        conv_layer.append(nn.ReLU())
        
        return nn.Sequential(*conv_layer)
    
    def forward(self, fused_img):
        output_1 = self.conv_layer_1(fused_img)
        output_2 = self.conv_layer_2(output_1)
        output_3 = self.conv_layer_3(torch.cat([output_1, output_2], dim=1))
        output_4 = self.conv_layer_4(torch.cat([output_1, output_2, output_3], dim=1))
        
        return torch.cat([output_1, output_2, output_3, output_4], dim=1) # channel: 16

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # Inter-Modality
        self.VisTransformerEncoder = nn.ModuleList([ViT() for _ in range(8)])
        self.IrTransformerEncoder = nn.ModuleList([ViT() for _ in range(8)])
        self.CrossTransformerEncoder =nn.ModuleList([CrossVit() for _ in range(8)])
        
        # Auxiliary 
        self.Auxiliary_Encoder = Auxiliary_Encoder()
    
    def forward(self, vis, ir):
        
        transformer_output = []
        for (VisTransformerEncoder, IrTransformerEncoder, CrossTransformerEncoder) in zip(self.VisTransformerEncoder, self.IrTransformerEncoder, self.CrossTransformerEncoder):
            VisTransformerEncoder_output = VisTransformerEncoder(vis)
            IrTransformerEncoder_output = IrTransformerEncoder(ir)
            
            CrossTransformerEncoder_output = CrossTransformerEncoder(VisTransformerEncoder_output, IrTransformerEncoder_output)
            transformer_output.append(CrossTransformerEncoder_output)
        
        concatenated_transformer_output = torch.cat(transformer_output, dim=1)
        auxiliary_output = self.Auxiliary_Encoder(torch.cat([vis, ir], dim=1))  # 16 channel

        output = torch.cat([concatenated_transformer_output, auxiliary_output], dim=1)
        
        return output

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.layer = nn.Sequential(
            self._conv_layer(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),
            self._conv_layer(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1),
            self._conv_layer(in_channels=8, out_channels=4, kernel_size=3, stride=1, padding=1),
            self._conv_layer(in_channels=4, out_channels=1, kernel_size=3, stride=1, padding=1, is_last=True)
        )
    
    def _conv_layer(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, is_last=False):
        if not is_last:
            activation = nn.ReLU()
        else:
            activation = nn.Tanh()
            
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            activation
        )    
    
    def forward(self, x):
        return self.layer(x)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.Encoder = Encoder()
        self.Decoder = Decoder()

    def forward(self, vis, ir):
        encoder_output = self.Encoder(vis, ir)
        decoder_output = self.Decoder(encoder_output) # fused output

        return decoder_output

def main():
    model = Generator()
    total_params = sum(p.numel() for p in model.parameters())
    summary(model, input_size=[(4, 1, 224, 224), (4, 1, 224, 224)])
    
if __name__ == "__main__":
    main()
        
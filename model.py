import torch
import torch.nn as nn

from einops import rearrange, repeat
from torchinfo import summary

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, emb_size: int = 768, img_size: int = 224):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.projection_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=emb_size, kernel_size=patch_size, stride=patch_size)
        )
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size)) # class token
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) ** 2 + 1, emb_size))
        
    def forward(self, x):
        b, _, _, _ = x.size()
        x = self.projection_conv(x)
        
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        print(cls_tokens.size())
        
def main():
    PE = PatchEmbedding()
    summary(PE, input_size=[(1, 1, 224, 224)])

if __name__ == "__main__":
    main()
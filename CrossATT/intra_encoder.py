import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchinfo import summary

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=1, patch_size=16, emb_size=256, img_size=224):
        super(PatchEmbedding, self).__init__()
        
        self.patch_size = patch_size
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e') # [batch, embedding, height, width] -> [batch, height*width, embedding]
        )
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) ** 2, emb_size)) # position embedding
    
    def forward(self, x):
        b, _, _, _ = x.size()
        x = self.projection(x) # patch size: 16x16 -> output size: 14x14
        
        # Class Token
        # cls_tokens = repeat(self.cls_tokens, '() n e -> b n e', b=b)
        # x = torch.cat([x, cls_tokens], dim=1)
        
        x += self.positions
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size=256, num_heads=8, dropout=0):
        super(MultiHeadAttention, self).__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        
        self.qkv = nn.Linear(emb_size, emb_size*3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
        
    def forward(self, x, mask=None):
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3) # [batch, 토큰 수, (헤드 수 * 각 헤드 차원 * 3)] -> [3, batch, 헤드 수, 토큰 수, 각 헤드 차원]
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        
        # scaled dot-product attention -> softmax((query*key)/sqrt(emb_size)) * value
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) # query, key의 내적
        scaling = self.emb_size ** (0.5) # query, key 내적 값 scaling
        att = F.softmax(energy / scaling, dim=-1) # query, key 내적 값 softmax
        att = self.att_drop(att)
        
        out = torch.einsum('bhal, bhlv -> bhav ', att, values) # softmax((query*key) / scaling) * values
        out = rearrange(out, "b h n d -> b n (h d)") # 모든 head의 차원을 합침 -> head가 나눠져있다가 합치는 것
        out = self.projection(out)
        
        return out
        
class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super(ResidualAdd, self).__init__()
        self.fn = fn
        
    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion=4, drop_p=0):
        super(FeedForwardBlock, self).__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size)
        )
        
class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, emb_size=768, drop_p=0, forward_expansion=4, forward_drop_p=0, **kwargs):
        super(TransformerEncoderBlock, self).__init__( # Sequential안에 Residual Add가 있다고 생각하면 됌
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            ))
        )

class TransformerEncoder(nn.Sequential):
    def __init__(self, depth=12, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])
        
class ViT(nn.Sequential):
    def __init__(self, in_channels=1, patch_size=16, emb_size=256, img_size=224, depth=3, **kwargs):
        super(ViT, self).__init__(
            PatchEmbedding(in_channels, patch_size, emb_size, img_size),
            TransformerEncoder(depth, emb_size=emb_size, **kwargs), # , 
        )

def main():
    model = ViT()
    summary(model, input_size=[(1, 1, 224, 224)])
    
if __name__ == "__main__":
    main()   
        
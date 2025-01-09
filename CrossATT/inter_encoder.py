import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, reduce, repeat
from torchinfo import summary

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size=256, num_heads=8, dropout=0):
        super(MultiHeadAttention, self).__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        
        self.vis_qkv = nn.Linear(emb_size, emb_size*3)
        self.ir_qkv = nn.Linear(emb_size, emb_size*3) 
        self.vis_projection = nn.Linear(emb_size, emb_size)
        self.ir_projection = nn.Linear(emb_size, emb_size)
        
    def _scaled_dot_attention(self, queries, keys, values, vis2ir=True):
        energy = torch.einsum("bhqd, bhkd -> bhqk", queries, keys)
        scaling = self.emb_size ** 0.5
        att = F.softmax(energy / scaling, dim=-1)
        
        out = torch.einsum("bhal, bhlv -> bhav", att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        
        if vis2ir:
            out = self.vis_projection(out)
        else:
            out = self.ir_projection(out)
        
        return out
    
    def forward(self, vis_emb, ir_emb):
        vis_qkv = rearrange(self.vis_qkv(vis_emb), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        ir_qkv = rearrange(self.ir_qkv(ir_emb), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)

        vis_queries, vis_keys, vis_values = vis_qkv[0], vis_qkv[1], vis_qkv[2]
        ir_queries, ir_keys, ir_values = ir_qkv[0], ir_qkv[1], ir_qkv[2]
        
        cross_vis2ir = self._scaled_dot_attention(vis_queries, ir_keys, ir_values, vis2ir=True)
        cross_ir2vis = self._scaled_dot_attention(ir_queries, vis_keys, vis_values, vis2ir=False)
        
        return cross_vis2ir, cross_ir2vis

class FeedForward(nn.Module):
    def __init__(self, in_channels, expansion=4, drop_p=0):
        super(FeedForward, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_channels, in_channels*expansion),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(in_channels*expansion, in_channels)
        )
    
    def forward(self, x):
        return self.layer(x)

class CrossVit(nn.Module):
    def __init__(self, emb_size=256):
        super(CrossVit, self).__init__()
        self.f_1_vis_norm = nn.LayerNorm(emb_size)
        self.f_1_ir_norm = nn.LayerNorm(emb_size)
        self.f_vis_norm = nn.LayerNorm(emb_size)
        self.f_ir_norm = nn.LayerNorm(emb_size)
        
        self.f_vis_feedforward = FeedForward(emb_size)
        self.f_ir_feedforward = FeedForward(emb_size)
        
        self.MultiHeadAttention = MultiHeadAttention()
    
    def _reconstruction(self, x):
        batch_size = 1
        x = x.view(batch_size, 14, 14, 16, 16)
        x = x.permute(0, 3, 4, 1, 2).contiguous() # [1, 16, 16, 14, 14]
        x = x.view(batch_size, 1, 224, 224)
        
        return x
    
    def forward(self, vis_emb, ir_emb):
        # Norm
        norm_vis_emb = self.f_1_vis_norm(vis_emb) # LN(Y_a)
        norm_ir_emb = self.f_1_ir_norm(ir_emb) # LN(Y_b)
        
        # Attention
        cross_vis2ir, cross_ir2vis = self.MultiHeadAttention(vis_emb, ir_emb) # # \hat Z_vis, \hat Z_ir
        
        # Residual Learning
        residual_vis2ir = norm_vis_emb + cross_vis2ir
        residual_ir2vis = norm_ir_emb + cross_ir2vis
        
        # LN / FFN / Residual Learning
        LN_residual_vis2ir = self.f_vis_norm(residual_vis2ir)
        LN_residual_ir2vis = self.f_ir_norm(residual_ir2vis)
        
        vis2ir_output = self.f_vis_feedforward(LN_residual_vis2ir) + LN_residual_vis2ir
        ir2vis_output = self.f_ir_feedforward(LN_residual_ir2vis) + LN_residual_ir2vis
        
        # Patch 복원
        reconstruction_vis2ir_output = self._reconstruction(vis2ir_output)
        reconstruction_ir2vis_output = self._reconstruction(ir2vis_output)
        
        return torch.cat([reconstruction_ir2vis_output, reconstruction_vis2ir_output], dim=1)
        
def main():
    model = CrossVit()
    # print(f"Img: {224*224}, Patch: {196*256}")
    summary(model=model, input_size=[(1, 196, 256), (1, 196, 256)])

if __name__ == "__main__":
    main()    
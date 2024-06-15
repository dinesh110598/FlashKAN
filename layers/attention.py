# %%
import torch
from torch import nn
from torch.nn import functional as F
from FlashKAN import FlashKAN

class MultiHeadKANAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.):
        assert (embed_dim % num_heads) == 0
        self.embed_dim = embed_dim
        self.nheads = num_heads
        
        self.Q_lin = FlashKAN(embed_dim, embed_dim, 32)
        self.K_lin = FlashKAN(embed_dim, embed_dim, 32)
        self.V_lin = FlashKAN(embed_dim, embed_dim, 32)
        self.drop = nn.Dropout(dropout)
        
    def forward(self, query, key, value):
        assert ((self.embed_dim==query.shape[-1]) and
                (self.embed_dim==key.shape[-1]))
        
        L = query.shape[-2]
        S = key.shape[-2]
        assert value.shape[-2]==S
        
        Q = self.Q_lin(query).reshape(
            *query.shape[:-1], self.nheads, -1).transpose(-3,-2)
        # (N..., H, L, E//H)
        K = self.K_lin(key).reshape(
            *key.shape[:-1], self.nheads, -1).transpose(-3,-2)
        # (N..., H, S, E//H)
        V = self.V_lin(value).reshape(
            *value.shape[:-1], self.nheads, -1).transpose(-3,-2)
        # (N..., H, S, E//H)
        
        attn_wt = self.drop(Q @ K.transpose(-2,-1)) # (N..., H, L, S)
        out = (attn_wt @ V).transpose(-3, -2) # (N..., L, H, E//H)
        
        return out.reshape(*out.shape[:-2], -1) # (N..., L, E)
        
class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, in_channels, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.conv = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.position_embeddings = nn.Embedding(patch_size**2, embed_dim)  # Learned positional embeddings

    def forward(self, x):
        # Assume x is a batch of images (shape: batch_size, in_channels, img_height, img_width)
        patches = self.conv(x).flatten(start_dim=1)  # Extract patches and flatten
        num_patches = patches.shape[1]  # Number of patches per image
        positions = torch.arange(0, num_patches, device=x.device).long()  # Patch positions
        embeddings = patches + self.position_embeddings(positions)  # Add positional embeddings
        return embeddings
    

class KANsformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=32, dropout=0.):
        super().__init__()
        
        self.self_attn = MultiHeadKANAttention(d_model, nhead)
        self.kan1 = FlashKAN(d_model, dim_feedforward, 64)
        self.kan2 = FlashKAN(dim_feedforward, d_model, 64)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, src, src_mask=None):
        src_norm = self.norm1(src)  # Apply layer normalization before self-attention
        attn_output, _ = self.self_attn(src_norm, src_norm, src_norm, attn_mask=src_mask)  # Self-attention
        src = src + self.dropout(attn_output)  # Add skip connection and dropout

        src_norm = self.norm2(src)  # Apply layer normalization before MLP
        src_2 = F.gelu(self.kan1(src_norm))  # MLP with GeLU activation
        src = src + self.dropout(self.kan2(src_2))  # Add skip connection and dropout

        return src
    
class VisionKANsformer(nn.Module):
    def __init__(self, patch_size, in_channels, embed_dim, num_heads, mlp_dim, num_layers):
        super(VisionKANsformer, self).__init__()
        self.patch_embedding = PatchEmbedding(patch_size, in_channels, embed_dim)
        self.transformer_encoder_classify = nn.Sequential(
            *(KANsformerEncoderLayer(embed_dim, num_heads, mlp_dim)
              for _ in range(num_layers)),
            FlashKAN(embed_dim, 1, 64),
            nn.Flatten(),
            FlashKAN(14*14, 10)
        )
        
    def forward(self, x):
        embeddings = self.patch_embedding(x)  # Get patch embeddings with positional info
        logits = self.transformer_encoder_classify(embeddings)  # Encode patch features
        # Add additional classification or regression head based on your task
        return logits
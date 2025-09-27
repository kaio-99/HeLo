import torch
import numpy as np
import torch.nn as nn

class FeedForward(nn.Module):
    """Feed forward neural network."""

    def __init__(self, model_dim, ffn_dim, dropout=0.0):
        super(FeedForward, self).__init__()
        self.w1 = nn.Linear(model_dim, ffn_dim)
        self.act = nn.GELU()
        self.w2 = nn.Linear(ffn_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        output = self.w2(self.act(self.w1(x)))
        output = self.dropout(output)
        # add residual and norm layer
        output = self.layer_norm(x + output)
        return output
    
class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism."""

    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None, mask_ratio=0.5, use_DropKey=True):

        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
            attention = attention * scale
        if attn_mask:
            attention = attention.masked_fill_(attn_mask, -np.inf)
        if use_DropKey:
            m_r = torch.ones_like(attention) * mask_ratio
            attention = attention + torch.bernoulli(m_r) * -1e12
        attention = self.softmax(attention)
        attention = self.dropout(attention)
        x = torch.bmm(attention, v)
        return x
    
class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""

    def __init__(self, model_dim, num_heads, attn_drop):
        super(MultiHeadAttention, self).__init__()

        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads, bias=False)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads, bias=False)
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads, bias=False)

        self.attention = ScaledDotProductAttention(attn_drop)
 
        # layer norm after multi-head attention
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x, y, attn_mask=None):
        B, C, T = x.shape
        # residual connection
        residual = x

        dim_per_head = self.dim_per_head
        num_heads = self.num_heads

        # linear projection
        query = self.linear_q(x).reshape(B, T, C)
        key = self.linear_k(y).reshape(B, T, C)
        value = self.linear_v(y).reshape(B, T, C)

        # split by heads
        key = key.view(B, -1, num_heads, dim_per_head).transpose(1, 2) \
            .reshape(B * num_heads, -1, dim_per_head)
        value = value.view(B, -1, num_heads, dim_per_head).transpose(1, 2) \
            .reshape(B * num_heads, -1, dim_per_head)
        query = query.view(B, -1, num_heads, dim_per_head).transpose(1, 2) \
            .reshape(B * num_heads, -1, dim_per_head)

        if attn_mask:
            attn_mask = attn_mask.repeat(num_heads, 1, 1)
        # scaled dot product attention
        scale = dim_per_head ** -0.5
        x = self.attention(
            query, key, value, scale, attn_mask)
        # concat heads
        x = x.view(B, num_heads, -1, dim_per_head).transpose(1, 2) \
            .reshape(B, -1, num_heads * dim_per_head)

        # add residual and norm layer
        x = self.layer_norm(residual + x)

        return x

class EncoderLayer(nn.Module):
    """Transformer layer."""

    def __init__(self, model_dim, num_heads, ffn_dim, attn_drop=0.0, feed_drop=0.0):
        super(EncoderLayer, self).__init__()

        self.attention = MultiHeadAttention(model_dim, num_heads, attn_drop)
        self.feed_forward = FeedForward(model_dim, ffn_dim, feed_drop)

    def forward(self, x, attn_mask=None):
        # self attention
        x = self.attention(x, x, attn_mask)
        # feed forward network
        x = self.feed_forward(x)

        return x

class TransformerEncoder(nn.Sequential):
    """Transformer encoder."""

    def __init__(self, depth, model_dim, num_heads, ffn_dim, attn_drop=0.0, feed_drop=0.0):
        super().__init__(*[EncoderLayer(model_dim, num_heads, ffn_dim, attn_drop, feed_drop) for _ in range(depth)])
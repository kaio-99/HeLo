import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .OT_torch_ import *
from .transformer_layers import *

class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism."""

    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, learned_correlation, scale=None, attn_mask=None, mask_ratio=0.5, use_DropKey=True):

        attention = torch.bmm(q, k.transpose(1, 2))

        attention = attention + learned_correlation
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
    
class LCD_CrossAttention(nn.Module):
    """Label Correlation Driven Cross Attention."""

    def __init__(self, model_dim, num_heads, attn_drop):
        super(LCD_CrossAttention, self).__init__()

        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads, bias=False)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads, bias=False)
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads, bias=False)
        self.attention = ScaledDotProductAttention(attn_drop)

        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x, y, learned_correlation, attn_mask=None):
        B, C, T = x.shape
        residual = x

        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        query = self.linear_q(x).reshape(B, T, C)
        key = self.linear_k(y).reshape(B, T, C)
        value = self.linear_v(y).reshape(B, T, C)

        key = key.view(B, -1, num_heads, dim_per_head).transpose(1, 2).reshape(B * num_heads, -1, dim_per_head)
        value = value.view(B, -1, num_heads, dim_per_head).transpose(1, 2).reshape(B * num_heads, -1, dim_per_head)
        query = query.view(B, -1, num_heads, dim_per_head).transpose(1, 2).reshape(B * num_heads, -1, dim_per_head)

        if attn_mask:
            attn_mask = attn_mask.repeat(num_heads, 1, 1)

        scale = dim_per_head ** -0.5
        x = self.attention(query, key, value, learned_correlation, scale, attn_mask)
        x = x.view(B, num_heads, -1, dim_per_head).transpose(1, 2).reshape(B, -1, num_heads * dim_per_head)

        x = self.layer_norm(residual + x)

        return x
    
class HeLo(nn.Module):
    """Proposed HeLo implementation."""

    def __init__(self, args):
        super(HeLo, self).__init__()

        self.emotion_classes = args.emotion_classes
        self.feature_dim = args.feature_dim
        self.n_channels = args.n_channels
        self.dim_eeg = args.dim_eeg
        self.dim_gsr = args.dim_gsr
        self.dim_ppg = args.dim_ppg
        self.dim_video = args.dim_video

        self.eeg_encoder = nn.Linear(args.dim_eeg, args.feature_dim)
        self.gsr_encoder = nn.Linear(args.dim_gsr, args.feature_dim * args.n_channels)
        self.ppg_encoder = nn.Linear(args.dim_ppg, args.feature_dim * args.n_channels)
        self.cross_attention = MultiHeadAttention(
            model_dim=args.feature_dim,
            num_heads=args.heads,
            attn_drop=args.dropout
        )
        self.projection = nn.Linear(args.feature_dim, args.feature_dim)
        self.behavioral_encoder = nn.Linear(args.dim_video, args.feature_dim * args.n_channels * 2)
        self.transformer_layer = TransformerEncoder(
            depth=args.depth,
            model_dim=args.feature_dim,
            num_heads=args.heads,
            ffn_dim=args.hidden_size,
            attn_drop=args.dropout,
            feed_drop=args.dropout,
        )
        
        self.linear = nn.Linear(1280, args.emotion_classes)
        self.fused_projection = nn.Linear(args.n_channels * 4, args.emotion_classes)

        self.lcd_cross_attention = LCD_CrossAttention(
            model_dim=args.feature_dim,
            num_heads=args.heads_lcd,
            attn_drop=args.dropout
        )

        self.Classifier = nn.Sequential(
            nn.Linear(1280, args.feature_dim),
            nn.ReLU(),
            nn.Linear(args.feature_dim, args.feature_dim),
            nn.ReLU(),
            nn.Linear(args.feature_dim, args.emotion_classes)
        )
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, eeg, gsr, ppg, video, labels):
        """
        :param eeg: eeg data, with dimension [B, 18, 5]
        :param gsr: gsr data, with dimension [B, 1, 28]
        :param ppg: ppg data, with dimension [B, 1, 27]
        :param video: video data, with dimension [B, 768]
        """
        label_embedding = nn.Parameter(torch.rand(eeg.shape[0], self.emotion_classes, self.feature_dim), requires_grad=True).cuda()
        eeg = self.eeg_encoder(eeg.reshape(eeg.shape[0], self.n_channels, self.dim_eeg))
        gsr = self.gsr_encoder(gsr.reshape(eeg.shape[0], 1, self.dim_gsr)).reshape(gsr.shape[0], self.n_channels, self.feature_dim)
        ppg = self.ppg_encoder(ppg.reshape(ppg.shape[0], 1, self.dim_ppg)).reshape(ppg.shape[0], self.n_channels, self.feature_dim)

        p1 = self.cross_attention(eeg, gsr)
        p2 = self.cross_attention(eeg, ppg)
        physiological = self.projection(torch.cat([p1, p2], dim=1)).reshape(p1.shape[0], self.n_channels * 2, self.feature_dim)
        behavioral = self.behavioral_encoder(video).reshape(video.shape[0], self.n_channels * 2, self.feature_dim)

        cost_distance = cost_matrix_batch_torch(physiological.transpose(1, 2), behavioral.transpose(1, 2))
        cost_distance = cost_distance.transpose(1, 2)

        flow, dist = GW_distance_uniform(physiological.transpose(1, 2), behavioral.transpose(1, 2))

        physiological = torch.matmul(flow, physiological)
        physiological = self.transformer_layer(physiological)
        behavioral = self.transformer_layer(behavioral)
        fused = torch.cat([physiological, behavioral], dim=1)
        learned_correlation, gt_correlation = self.label_correlation(label_embedding, labels.reshape(labels.shape[0], self.emotion_classes, 1))

        fused = self.fused_projection(fused.transpose(1, 2)).transpose(1, 2)
        out = self.lcd_cross_attention(label_embedding, fused, learned_correlation)

        predict = self.Classifier(torch.flatten(out, start_dim=1))
        predict = self.softmax(predict)

        return predict, learned_correlation, gt_correlation
    
    def label_correlation(self, features, labels):
        labels = F.normalize(labels)
        gt_correlation = torch.matmul(labels, labels.transpose(1, 2))

        features = F.normalize(features)
        learned_correlation = torch.matmul(features, features.transpose(1, 2))

        return learned_correlation, gt_correlation
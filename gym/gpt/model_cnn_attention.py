import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class SpatialAttention(nn.Module):
    """Spatial attention module"""
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        attention = self.conv(x)
        attention = self.sigmoid(attention)
        return x * attention


class ChannelAttention(nn.Module):
    """Channel attention module (Squeeze-and-Excitation)"""
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class AttentionBlock(nn.Module):
    """Combined channel and spatial attention"""
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.channel_attention = ChannelAttention(in_channels)
        self.spatial_attention = SpatialAttention(in_channels)
    
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class RewardModelMinkowski(nn.Module):
    def __init__(self, drone_info_dim=46, latent_dim=512):
        super(RewardModelMinkowski, self).__init__()

        # Attention이 추가된 인코더
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            AttentionBlock(64),
            nn.Dropout2d(0.1),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            AttentionBlock(128),
            nn.Dropout2d(0.1),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            AttentionBlock(256),
            nn.Dropout2d(0.1),
            
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            AttentionBlock(512),
            nn.Dropout2d(0.1),
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc_enc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, drone_info, obs):
        x = self.encoder(obs)
        x_avg = self.global_pool(x).flatten(1)
        x_max = self.global_max_pool(x).flatten(1)
        x_combined = torch.cat([x_avg, x_max], dim=-1)
        logits = self.fc_enc(x_combined)
        return logits

    def get_trainable_parameters(self):
        return self.parameters()
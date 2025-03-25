import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

torch.set_printoptions(threshold=float('inf'), linewidth=10000)  # 무한대 대신 큰 정수 사용
np.set_printoptions(threshold=np.inf, linewidth=10000)  # 무한대 대신 큰 정수 사용


class RewardModelCombined(nn.Module):
    def __init__(self, obstacle_encoder, path_encoder, drone_info_dim=46, latent_dim=128, dropout_rate=0.2):
        super(RewardModelCombined, self).__init__()

        # 오토인코더에서 인코더 부분만 사용
        self.obstacle_encoder = obstacle_encoder
        for param in self.obstacle_encoder.parameters():
            param.requires_grad = False  # 사전 학습된 가중치 고정
            
        self.path_encoder = path_encoder
        for param in self.path_encoder.parameters():
            param.requires_grad = False  # 사전 학습된 가중치 고정

        self.drone_info_dim = nn.Sequential(
            nn.Linear(drone_info_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
        )

        # 드론 정보 인코더
        self.info_path_encoder = nn.Sequential(
            nn.Linear(128 + 128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
        )

        self.obs_path_encoder = nn.Sequential(
            nn.Linear(128 + 128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
        )

        # 결합 및 보상 예측 레이어
        self.reward_predictor = nn.Sequential(
            nn.Linear(64 + 64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout(0.1)
        self.batch_norm = nn.BatchNorm1d(128)

    def forward(self, drone_info, obs, path):
        # 각 입력 처리
        self.obstacle_encoder.eval()
        self.path_encoder.eval()
        obs_features = self.obstacle_encoder.encoder(obs)
        obs_features = torch.flatten(obs_features, start_dim=1)
        obs_features = self.obstacle_encoder.fc_enc(obs_features)

        path_features = self.path_encoder.encoder(path)
        path_features = torch.flatten(path_features, start_dim=1)
        path_features = self.path_encoder.fc_enc(path_features)

        drone_info_features = self.drone_info_dim(drone_info)
        
        # 특성 결합
        obs_path_features = torch.cat([obs_features, path_features], dim=-1)
        obs_path_features = self.obs_path_encoder(obs_path_features)

        info_path_features = torch.cat([drone_info_features, path_features], dim=-1)
        info_path_features = self.info_path_encoder(info_path_features)

        combined_features = torch.cat([obs_path_features, info_path_features], dim=-1)
        # 보상 예측
        reward = self.reward_predictor(combined_features)
        
        return reward
    
    def get_trainable_parameters(self):
        """학습 가능한 파라미터만 반환하는 메서드"""
        return list(self.info_path_encoder.parameters()) + list(self.reward_predictor.parameters()) + list(self.obs_path_encoder.parameters())
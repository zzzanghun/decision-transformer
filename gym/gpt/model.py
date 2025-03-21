import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class RewardModel(nn.Module):
    def __init__(self, obstacle_encoder, path_encoder, drone_info_dim=46, latent_dim=128, dropout_rate=0.2):
        super(RewardModel, self).__init__()

        # 오토인코더에서 인코더 부분만 사용
        self.obstacle_encoder = obstacle_encoder
        for param in self.obstacle_encoder.parameters():
            param.requires_grad = False  # 사전 학습된 가중치 고정
            
        self.path_encoder = path_encoder
        for param in self.path_encoder.parameters():
            param.requires_grad = False  # 사전 학습된 가중치 고정

        # BatchNorm 레이어를 모델 초기화 시 생성
        self.obs_batch_norm = nn.BatchNorm1d(128 * 13 * 13)
        self.path_batch_norm = nn.BatchNorm1d(128 * 13 * 13)

        # 드론 정보 인코더 (더 작은 모델로 변경)
        self.drone_info_encoder = nn.Sequential(
            nn.Linear(drone_info_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128)
        )

        # 결합 및 보상 예측 레이어 (더 작은 모델로 변경)
        self.reward_predictor = nn.Sequential(
            nn.Linear(128 + 128 + 128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, drone_info, obs, path):
        # 각 입력 처리
        obs_features = self.obstacle_encoder.encoder(obs)
        obs_features = torch.flatten(obs_features, start_dim=1)
        obs_features = self.dropout(obs_features)
        obs_features = self.obs_batch_norm(obs_features)  # 미리 생성된 BatchNorm 사용
        obs_features = self.obstacle_encoder.fc_enc(obs_features)
        
        path_features = self.path_encoder.encoder(path)
        path_features = torch.flatten(path_features, start_dim=1)
        path_features = self.dropout(path_features)
        path_features = self.path_batch_norm(path_features)  # 미리 생성된 BatchNorm 사용
        path_features = self.path_encoder.fc_enc(path_features)
        
        drone_features = self.drone_info_encoder(drone_info)
        
        # 특성 결합
        combined_features = torch.cat([obs_features, path_features, drone_features], dim=1)
        
        # 보상 예측
        reward = self.reward_predictor(combined_features)
        
        return reward

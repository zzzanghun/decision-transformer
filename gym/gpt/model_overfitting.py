import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

torch.set_printoptions(threshold=float('inf'), linewidth=10000)  # 무한대 대신 큰 정수 사용
np.set_printoptions(threshold=np.inf, linewidth=10000)  # 무한대 대신 큰 정수 사용


class RewardModelOverfitting(nn.Module):
    def __init__(self, obstacle_encoder, path_encoder, drone_info_dim=46, latent_dim=128):
        super(RewardModelOverfitting, self).__init__()

        # 오토인코더에서 인코더 부분만 사용
        self.obstacle_encoder = obstacle_encoder
        for param in self.obstacle_encoder.parameters():
            param.requires_grad = False  # 사전 학습된 가중치 고정
            
        self.path_encoder = path_encoder
        for param in self.path_encoder.parameters():
            param.requires_grad = False  # 사전 학습된 가중치 고정

        # 드론 정보 인코더
        self.drone_info_encoder = nn.Sequential(
            nn.Linear(drone_info_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128)
        )

        # 결합 및 보상 예측 레이어
        self.reward_predictor = nn.Sequential(
            nn.Linear(128 + 128 + 128, 512),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.3),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.3),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.3),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
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

        # obs_before = obs[0].detach().cpu().numpy()
        # print(obs_before.astype(int), "before")
        # obs_decoder = self.obstacle_encoder.fc_dec(obs_features)
        # obs_decoder = obs_decoder.view(obs_decoder.size(0), 128, 13, 13)
        # obs_decoder = self.obstacle_encoder.decoder(obs_decoder)
        # obs_after = obs_decoder[0].detach().cpu().numpy()
        # print(obs_after.astype(int), "after")
        # differences = np.not_equal(obs_before, obs_after)
        # num_differences = np.sum(differences)
        # print(f"다른 요소의 개수: {num_differences} / 10000 ({(num_differences/10000)*100:.2f}%)")
        # print("\n차이가 있는 위치 (1은 차이가 있는 위치):")
        # print(differences.astype(int))

        path_features = self.path_encoder.encoder(path)
        path_features = torch.flatten(path_features, start_dim=1)
        path_features = self.path_encoder.fc_enc(path_features)

        # path_before = path[0].detach().cpu().numpy()
        # print(path_before.astype(int), "before")
        # path_decoder = self.path_encoder.fc_dec(path_features)
        # path_decoder = path_decoder.view(path_decoder.size(0), 128, 13, 13)
        # path_decoder = self.path_encoder.decoder(path_decoder)
        # path_after = path_decoder[0].detach().cpu().numpy()
        # print(path_after.astype(int), "after")
        # differences = np.not_equal(path_before, path_after)
        # num_differences = np.sum(differences)
        # print(f"다른 요소의 개수: {num_differences} / 10000 ({(num_differences/10000)*100:.2f}%)")
        # print("\n차이가 있는 위치 (1은 차이가 있는 위치):")
        # print(differences.astype(int))

        drone_features = self.drone_info_encoder(drone_info)
        
        # 특성 결합
        combined_features = torch.cat([obs_features, path_features, drone_features], dim=-1)
        
        # 보상 예측
        reward = self.reward_predictor(combined_features)
        
        return reward
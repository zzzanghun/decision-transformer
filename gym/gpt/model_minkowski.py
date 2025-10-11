import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import MinkowskiEngine as ME

torch.set_printoptions(threshold=float('inf'), linewidth=10000)  # 무한대 대신 큰 정수 사용
np.set_printoptions(threshold=np.inf, linewidth=10000)  # 무한대 대신 큰 정수 사용


class RewardModelMinkowski(nn.Module):
    def __init__(self, drone_info_dim=46, latent_dim=128):
        super(RewardModelMinkowski, self).__init__()

        # 2D MinkowskiEngine 인코더
        self.encoder = nn.Sequential(
            ME.MinkowskiConvolution(1, 32, kernel_size=3, stride=1, dimension=2),
            # ME.MinkowskiBatchNorm(32),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiDropout(0.1),

            ME.MinkowskiConvolution(32, 64, kernel_size=3, stride=2, dimension=2),
            # ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiDropout(0.1),

            ME.MinkowskiConvolution(64, 128, kernel_size=3, stride=2, dimension=2),
            # ME.MinkowskiBatchNorm(128),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiDropout(0.1),

            ME.MinkowskiConvolution(128, 256, kernel_size=3, stride=1, dimension=2),
            # ME.MinkowskiBatchNorm(256),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiDropout(0.1),
        )

        self.global_pool = ME.MinkowskiGlobalAvgPooling()
        self.norm_global_pool = nn.LayerNorm(256)
        self.fc_enc = nn.Linear(256, 128)

        # 드론 정보 인코더
        self.drone_info_encoder = nn.Sequential(
            nn.Linear(drone_info_dim, 256),
            nn.GELU(),
            nn.Linear(256, 128)
        )

        self.drone_info_norm = nn.LayerNorm(128)
        self.obs_norm = nn.LayerNorm(128)

        # 결합 및 보상 예측 레이어
        self.reward_predictor = nn.Sequential(
            nn.Linear(128 + 128, 512),
            nn.GELU(),

            nn.Linear(512, 256),
            nn.GELU(),

            nn.Linear(256, 128),
            nn.GELU(),

            nn.Linear(128, 64),
            nn.GELU(),

            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout(0.1)
        self.batch_norm = nn.BatchNorm1d(128)

    def forward(self, drone_info, obs):
        """
        Args:
            drone_info: (batch_size, drone_info_dim) - 드론 정보
            obs: list of (coords, feats) for each batch item - 2D projection된 장애물 포인트클라우드

        Returns:
            reward: (batch_size, 1) - 예측된 보상
        """
        batch_size = drone_info.shape[0]

        # 배치의 모든 장애물 데이터 수집
        coords_list, feats_list = [], []

        for b in range(batch_size):
            # obs[b]는 (coords, feats) 튜플
            coords = obs[b][0]  # 좌표 (N_points, 3) - [batch_idx, y, x] (2D)
            feats = obs[b][1]   # 특성 (N_points, 1)

            # 배치 인덱스 설정 (중요: MinkowskiEngine에서 배치를 구분하기 위함)
            coords = coords.clone()
            coords[:, 0] = int(b)
            coords_list.append(coords)
            feats_list.append(feats)

        combined_coords = torch.cat(coords_list, dim=0)
        combined_feats = torch.cat(feats_list, dim=0)

        # MinkowskiEngine 스파스 텐서 생성
        sparse_tensor = ME.SparseTensor(
            features=combined_feats,
            coordinates=combined_coords,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )

        # 인코더 네트워크 통과
        x = self.encoder(sparse_tensor)

        # 글로벌 풀링으로 각 배치 항목을 고정 크기 벡터로 변환
        x = self.global_pool(x)
        x = F.gelu(x.F)

        # 최종 임베딩 생성 (batch_size, 128)
        obstacles_embeddings = self.fc_enc(x)

        assert obstacles_embeddings.shape[0] == batch_size, \
            f"Expected batch_size {batch_size}, got {obstacles_embeddings.shape[0]}"

        # 드론 정보 인코딩 (batch_size, 128)
        drone_info_features = self.drone_info_encoder(drone_info)

        obstacles_embeddings = self.obs_norm(obstacles_embeddings)
        drone_info_features = self.drone_info_norm(drone_info_features)

        drone_info_features = self.dropout(drone_info_features)

        # 특성 결합 (batch_size, 128 + 128)
        combined_features = torch.cat([obstacles_embeddings, drone_info_features], dim=-1)

        # 보상 예측 (batch_size, 1)
        reward = self.reward_predictor(combined_features)

        return reward
    
    def get_trainable_parameters(self):
        """학습 가능한 파라미터만 반환하는 메서드"""
        return list(self.drone_info_encoder.parameters()) + list(self.reward_predictor.parameters())

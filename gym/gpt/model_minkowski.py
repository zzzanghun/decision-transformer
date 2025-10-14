import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import MinkowskiEngine as ME

torch.set_printoptions(threshold=float('inf'), linewidth=10000)  # 무한대 대신 큰 정수 사용
np.set_printoptions(threshold=np.inf, linewidth=10000)  # 무한대 대신 큰 정수 사용


class RewardModelMinkowski(nn.Module):
    def __init__(self, drone_info_dim=46, latent_dim=256):
        super(RewardModelMinkowski, self).__init__()

        # 2D MinkowskiEngine 인코더 - BatchNorm 활성화하고 더 깊게
        self.encoder = nn.Sequential(
            ME.MinkowskiConvolution(1, 64, kernel_size=3, stride=1, dimension=2),
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU(inplace=True),

            ME.MinkowskiConvolution(64, 128, kernel_size=3, stride=2, dimension=2),
            ME.MinkowskiBatchNorm(128),
            ME.MinkowskiReLU(inplace=True),

            # ME.MinkowskiConvolution(128, 256, kernel_size=3, stride=2, dimension=2),
            # ME.MinkowskiBatchNorm(256),
            # ME.MinkowskiReLU(inplace=True),

            # ME.MinkowskiConvolution(256, 512, kernel_size=3, stride=2, dimension=2),
            # ME.MinkowskiBatchNorm(512),
            # ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiDropout(0.1),  # 마지막에만 최소 dropout
        )

        self.global_pool = ME.MinkowskiGlobalAvgPooling()
        self.global_max_pool = ME.MinkowskiGlobalMaxPooling()

        # Global pooling 후 결합 (avg + max pooling)
        self.fc_enc = nn.Sequential(
            nn.Linear(256, 256),  # 512 (avg) + 512 (max) = 1024
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, latent_dim)
        )
        self.obs_norm = nn.LayerNorm(latent_dim)

        # 드론 정보 인코더 - 더 강력하게
        self.drone_info_encoder = nn.Sequential(
            nn.Linear(drone_info_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, latent_dim),
            # nn.ReLU(),
            # nn.Dropout(0.05),
            # nn.Linear(384, latent_dim)
        )
        self.drone_info_norm = nn.LayerNorm(latent_dim)

        # 결합 및 이진 분류 예측 레이어 - 더 깊고 강력하게
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
            # nn.ReLU(),
            # nn.Dropout(0.1),
            # nn.Linear(384, 256),
            # nn.ReLU(),
            # nn.Linear(256, 128),
            # nn.ReLU(),
            # nn.Linear(128, 1)  # Sigmoid는 loss function에서 처리
        )

    def forward(self, drone_info, obs):
        """
        Args:
            drone_info: (batch_size, drone_info_dim) - 드론 정보
            obs: list of (coords, feats) for each batch item - 2D projection된 장애물 포인트클라우드
                 coords는 이미 배치 인덱스가 포함된 (N_points, 3) - [batch_idx, y, x] 형태

        Returns:
            logits: (batch_size, 1) - 이진 분류 로짓 (0: 안전, 1: 효율)
        """
        batch_size = drone_info.shape[0]

        # 배치의 모든 장애물 데이터 수집
        coords_list, feats_list = [], []

        for b in range(batch_size):
            # obs[b]는 (coords, feats) 튜플
            coords = obs[b][0]  # 좌표 (N_points, 3) - [batch_idx, y, x] (2D) - 이미 배치 인덱스 포함
            feats = obs[b][1]   # 특성 (N_points, 1)

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

        # 글로벌 풀링으로 각 배치 항목을 고정 크기 벡터로 변환 (avg + max)
        x_avg = self.global_pool(x)
        x_max = self.global_max_pool(x)

        # Avg와 Max pooling 결과 결합
        x_combined = torch.cat([x_avg.F, x_max.F], dim=-1)

        # 최종 임베딩 생성 (batch_size, latent_dim)
        obstacles_embeddings = self.fc_enc(x_combined)
        obstacles_embeddings = self.obs_norm(obstacles_embeddings)

        assert obstacles_embeddings.shape[0] == batch_size, \
            f"Expected batch_size {batch_size}, got {obstacles_embeddings.shape[0]}"

        # 드론 정보 인코딩 (batch_size, latent_dim)
        drone_info_features = self.drone_info_encoder(drone_info)
        drone_info_features = self.drone_info_norm(drone_info_features)

        # 특성 결합 (batch_size, latent_dim * 2)
        combined_features = torch.cat([obstacles_embeddings, drone_info_features], dim=-1)

        # 이진 분류 예측 (batch_size, 1)
        logits = self.classifier(combined_features)

        return logits

    def get_trainable_parameters(self):
        """학습 가능한 파라미터만 반환하는 메서드 (전체 모델 학습)"""
        return self.parameters()

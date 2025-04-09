import torch
import torch.nn as nn
import MinkowskiEngine as ME

class SparseVoxelAutoencoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(SparseVoxelAutoencoder, self).__init__()

        # Encoder 정의
        self.encoder = nn.Sequential(
            ME.MinkowskiConvolution(1, 32, kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(32),
            ME.MinkowskiReLU(inplace=True),

            ME.MinkowskiConvolution(32, 64, kernel_size=3, stride=2, dimension=3),
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU(inplace=True),

            ME.MinkowskiConvolution(64, 128, kernel_size=3, stride=2, dimension=3),
            ME.MinkowskiBatchNorm(128),
            ME.MinkowskiReLU(inplace=True)
        )

        # Global pooling 추가
        self.global_pool = ME.MinkowskiGlobalAvgPooling()

        # latent_dim 벡터로 압축 및 복원 (dense linear 사용)
        self.fc_enc = nn.Linear(128, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, 128)

        # Decoder 정의 (기존과 동일)
        self.decoder = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(128, 128, kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(128),
            ME.MinkowskiReLU(inplace=True),

            ME.MinkowskiConvolutionTranspose(128, 64, kernel_size=3, stride=2, dimension=3),
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU(inplace=True),

            ME.MinkowskiConvolutionTranspose(64, 32, kernel_size=3, stride=2, dimension=3),
            ME.MinkowskiBatchNorm(32),
            ME.MinkowskiReLU(inplace=True),

            ME.MinkowskiConvolution(32, 1, kernel_size=1, stride=1, dimension=3),
            ME.MinkowskiSigmoid()
        )

    def forward(self, coordinates, features):
        # Minkowski sparse tensor 생성
        x = ME.SparseTensor(features, coordinates)

        # Encoder 적용
        encoded = self.encoder(x)

        # print(encoded.F.shape, "encoded.F.shape")

        # Global Pooling을 통해 (batch_size, feature) 생성
        pooled = self.global_pool(encoded)
        latent = self.fc_enc(pooled.F)  # latent shape: (batch_size, latent_dim)

        # latent를 다시 sparse tensor의 각 voxel로 broadcast
        batch_indices = encoded.C[:, 0]  # 각 voxel이 속한 batch index 추출
        expanded_latent = latent[batch_indices]  # latent 벡터를 voxel 개수로 확장

        decoded_features = self.fc_dec(expanded_latent)  # 원래 voxel 수와 동일해짐!

        decoded_sparse = ME.SparseTensor(
            features=decoded_features,
            coordinate_map_key=encoded.coordinate_map_key,
            coordinate_manager=encoded.coordinate_manager
        )

        # Decoder를 통해 원본 크기로 복원
        reconstructed = self.decoder(decoded_sparse)

        return reconstructed, latent  # latent shape: (batch_size, 128)

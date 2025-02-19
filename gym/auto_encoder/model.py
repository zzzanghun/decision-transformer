import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class CostmapConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(CostmapConvAutoencoder, self).__init__()

        ########################################
        #            ENCODER PART             #
        ########################################
        self.encoder = nn.Sequential(
            # [1, 100, 100] -> [16, 100, 100] (stride=1)
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),

            # [16, 100, 100] -> [16, 50, 50] -> [32, 50, 50] (stride=2)
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1, groups=16, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),

            # [32, 50, 50] -> [32, 25, 25] -> [64, 25, 25] (stride=2)
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, groups=32, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),

            # [64, 25, 25] -> [64, 13, 13] -> [128, 13, 13] (stride=2)
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, groups=64, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True)
        )

        # 최종 [128, 13, 13] 특징맵을 latent_dim 차원으로 압축
        self.fc_enc = nn.Linear(128 * 13 * 13, latent_dim)

        # latent_dim -> [128, 13, 13]으로 복원
        self.fc_dec = nn.Linear(latent_dim, 128 * 13 * 13)

        ########################################
        #            DECODER PART             #
        ########################################
        # 목표: (13->25->50->100)
        self.decoder = nn.Sequential(
            # (128,13,13) -> (64,25,25)
            nn.ConvTranspose2d(
                in_channels=128, out_channels=64,
                kernel_size=3, stride=2, padding=1, output_padding=0,  # 여기서 output_padding=0
                bias=False
            ),
            nn.ReLU(inplace=True),

            # (64,25,25) -> (32,50,50)
            nn.ConvTranspose2d(
                in_channels=64, out_channels=32,
                kernel_size=3, stride=2, padding=1, output_padding=1,  # output_padding=1
                bias=False
            ),
            nn.ReLU(inplace=True),

            # (32,50,50) -> (16,100,100)
            nn.ConvTranspose2d(
                in_channels=32, out_channels=16,
                kernel_size=3, stride=2, padding=1, output_padding=1,  # output_padding=1
                bias=False
            ),
            nn.ReLU(inplace=True),

            # 채널 축소: [16, 100, 100] -> [1, 100, 100]
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3,
                      stride=1, padding=1, bias=True),
            nn.Sigmoid()  # (0~1) 범위로 복원
        )

    def forward(self, x):
        # Encoder
        x = self.encoder(x)                  # [batch_size, 128, 13, 13]
        x = nn.Flatten(start_dim=1)(x)       # [batch_size, 128*13*13]
        x = self.fc_enc(x)                   # [batch_size, latent_dim]

        # Decoder
        x = self.fc_dec(x)                   # [batch_size, 128*13*13]
        x = x.view(x.size(0), 128, 13, 13)   # [batch_size, 128, 13, 13]
        x = self.decoder(x)                  # [batch_size, 1, 100, 100]
        return x

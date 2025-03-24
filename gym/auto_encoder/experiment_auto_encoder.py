import torch
import torch.nn as nn
import torch.optim as optim
from data import CostmapDataset
from model import CostmapConvAutoencoder
from torch.utils.data import DataLoader, Dataset
import wandb
import os
import numpy as np

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
print(PROJECT_PATH)

class RandomCostmapDataset(Dataset):
    """
    0, 0.5, 1 값을 무작위로 배치한 코스트맵 데이터셋
    """
    def __init__(self, size=100, num_samples=50000):
        """
        size: 코스트맵의 크기 (size x size)
        num_samples: 데이터셋의 샘플 수
        """
        self.size = size
        self.num_samples = num_samples
        self.data = self._generate_random_data()
        
    def _generate_random_data(self):
        # 0, 0.5, 1 값을 가진 랜덤 데이터 생성
        data = []
        for _ in range(self.num_samples):
            # 먼저 모든 값을 0으로 초기화
            costmap = np.zeros((1, self.size, self.size), dtype=np.float32)
            
            # 0.5 값을 가질 픽셀 수 (전체 픽셀의 10%)
            num_half_pixels = int(0.1 * self.size * self.size)
            # 1 값을 가질 픽셀 수 (전체 픽셀의 5%)
            num_one_pixels = int(0.2 * self.size * self.size)
            
            # 0.5 값 랜덤 배치
            half_indices = np.random.choice(self.size * self.size, num_half_pixels, replace=False)
            for idx in half_indices:
                row, col = idx // self.size, idx % self.size
                costmap[0, row, col] = 0.5
            
            # 1 값 랜덤 배치
            one_indices = np.random.choice(self.size * self.size, num_one_pixels, replace=False)
            for idx in one_indices:
                row, col = idx // self.size, idx % self.size
                costmap[0, row, col] = 1.0
                
            data.append(torch.tensor(costmap, dtype=torch.float32))
            
        return data
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx]

def weighted_mse_loss(output, target):
    # output, target shape = (batch, 1, H, W)
    weight = torch.ones_like(target)
    weight[target == 0.5] = 10.0  # 0.5인 지점의 손실 가중치 크게
    return (weight * (output - target)**2).mean()
    
def train_autoencoder(model, dataloader, epochs=10, lr=1e-3):
    """
    오토인코더 학습 루틴 예시.
    - dataloader: CostmapDataset 등에 대한 DataLoader
    - epochs: 학습 epoch 수
    - lr: 학습률
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # binary map이라면 BCEWithLogitsLoss나 BCELoss 고려
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for data in dataloader:
            # data shape: [batch_size, 1, 100, 100]
            data = data.to(device)
            
            optimizer.zero_grad()
            
            # Forward
            outputs = model(data)
            
            # Loss 계산
            # loss = criterion(outputs, data)
            loss = weighted_mse_loss(outputs, data)

            # Backprop
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * data.size(0)
        
        epoch_loss = running_loss / len(dataloader.dataset)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")

        wandb.log({
            "loss": epoch_loss
        })

        if epoch % 100 == 0:
            folder_name = f"{PROJECT_PATH}/model/auto_encoder"
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
            torch.save(model.state_dict(), f"{PROJECT_PATH}/model/auto_encoder/autoencoder_with_runlength_{epoch}.pth")


if __name__ == "__main__":
    # 1) 데이터셋 및 데이터로더 준비
    dataset = CostmapDataset(add_random_lines=False)  # 임의 생성 예시
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # 2) 모델 생성
    model = CostmapConvAutoencoder(latent_dim=128)
    model.load_state_dict(torch.load(f"{PROJECT_PATH}/model/model_900.pth"))
    
    wandb.init(
            project='auto-encoder'
        )

    # 3) 학습
    train_autoencoder(model, dataloader, epochs=1000000, lr=1e-4)
    
    # 4) 추론(테스트) 예시
    # 실제 테스트 시에는 별도의 검증 세트를 사용해야 함
    sample_data = next(iter(dataloader))
    with torch.no_grad():
        sample_data = sample_data.to(next(model.parameters()).device)
        reconstructed = model(sample_data)
    
    print("Input shape:", sample_data.shape)        # [batch, 1, 100, 100]
    print("Output shape:", reconstructed.shape)     # [batch, 1, 100, 100]

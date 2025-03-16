import torch
import torch.nn as nn
import torch.optim as optim
from data import CostmapDataset
from model import CostmapConvAutoencoder
from torch.utils.data import DataLoader
import wandb
import os

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
print(PROJECT_PATH)
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
            loss = criterion(outputs, data)
            
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
            torch.save(model.state_dict(), f"{PROJECT_PATH}/model/auto_encoder/model_{epoch}.pth")


if __name__ == "__main__":
    # 1) 데이터셋 및 데이터로더 준비
    dataset = CostmapDataset()  # 임의 생성 예시
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # 2) 모델 생성
    model = CostmapConvAutoencoder(latent_dim=128)
    
    wandb.init(
            project='auto-encoder'
        )

    # 3) 학습
    train_autoencoder(model, dataloader, epochs=1000000, lr=1e-3)
    
    # 4) 추론(테스트) 예시
    # 실제 테스트 시에는 별도의 검증 세트를 사용해야 함
    sample_data = next(iter(dataloader))
    with torch.no_grad():
        sample_data = sample_data.to(next(model.parameters()).device)
        reconstructed = model(sample_data)
    
    print("Input shape:", sample_data.shape)        # [batch, 1, 100, 100]
    print("Output shape:", reconstructed.shape)     # [batch, 1, 100, 100]

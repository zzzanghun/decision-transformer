import torch
import torch.nn as nn
import torch.optim as optim
from data import CostmapDataset
from model import CostmapConvAutoencoder
from torch.utils.data import DataLoader
import wandb
import os
import numpy as np

np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)


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
            folder_name = f"/home/zzzanghun/git/decision-transformer/gym/model/auto_encoder"
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
            torch.save(model.state_dict(), f"/home/zzzanghun/git/decision-transformer/gym/model/auto_encoder/model_{epoch}.pth")


if __name__ == "__main__":
    # 1) 데이터셋 및 데이터로더 준비
    dataset = CostmapDataset()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # 2) 모델 생성
    model = CostmapConvAutoencoder(latent_dim=128)
    model.load_state_dict(torch.load(f"/home/zzzanghun/git/decision-transformer/gym/model/auto_encoder/model_900.pth"))
    model.eval()    

    sample_data = next(iter(dataloader))

    print(sample_data.shape)

    with torch.no_grad():
        sample_data = sample_data.to(next(model.parameters()).device)
        reconstructed = model(sample_data)
    
    print("Input shape:", sample_data.shape)
    print("Output shape:", reconstructed.shape)

    # 0.5를 기준으로 이진화
    reconstructed_binary = (reconstructed > 0.9).float()
    
    # 2D matrix로 변환 (배치 차원과 채널 차원 제거)
    sample_2d = sample_data.squeeze().cpu().numpy()
    reconstructed_2d = reconstructed_binary.squeeze().cpu().numpy()
    
    # 차이 계산
    differences = np.not_equal(sample_2d, reconstructed_2d)
    num_differences = np.sum(differences)
    
    print("\n원본 데이터:")
    print(sample_2d)
    print("\n재구성된 데이터 (이진화):")
    print(reconstructed_2d)
    print(f"\n다른 요소의 개수: {num_differences} / 10000 ({(num_differences/10000)*100:.2f}%)")
    
    # 차이가 있는 위치 시각화 (선택적)
    print("\n차이가 있는 위치 (1은 차이가 있는 위치):")
    print(differences.astype(int))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data import CostmapDataset
from model import SparseVoxelAutoencoder
import wandb
import os
import MinkowskiEngine as ME

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def minkowski_collate_fn(batch):
    coordinates, features = [], []
    for i, item in enumerate(batch):
        coords = item['coordinates']
        coords[:, 0] = i  # batch index 설정
        coordinates.append(coords)
        features.append(item['features'])

    coordinates = torch.cat(coordinates, dim=0)
    features = torch.cat(features, dim=0)
    return coordinates, features

def sparse_mse_loss(output_sparse_tensor, target_sparse_tensor):
    return torch.nn.functional.mse_loss(output_sparse_tensor.F, target_sparse_tensor.F)

def reconstruction_accuracy(output_sparse_tensor, target_sparse_tensor, threshold=0.5):
    # 출력과 타겟의 좌표 가져오기
    out_coords = output_sparse_tensor.C.cpu().numpy()
    target_coords = target_sparse_tensor.C.cpu().numpy()
    
    # 좌표 일치 여부 확인 (좌표 매니저가 동일하다면 좌표 순서도 동일해야 함)
    if not torch.equal(output_sparse_tensor.C, target_sparse_tensor.C):
        print("경고: 출력과 타겟 텐서의 좌표가 일치하지 않습니다.")
    
    # 이진화 변환
    output_binary = (output_sparse_tensor.F > threshold).float()
    target_binary = (target_sparse_tensor.F > threshold).float()

    true_positives = ((output_binary == 1) & (target_binary == 1)).float().sum()
    total_actual_positives = (target_binary == 1).float().sum()

    recall = true_positives / total_actual_positives
    
    return recall

def train_autoencoder(model, dataloader, epochs=10, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_accuracy = 0.0

        for coordinates, features in dataloader:
            coordinates = coordinates.to(device)
            features = features.to(device)

            optimizer.zero_grad()

            output_sparse, latent = model(coordinates, features)

            target_sparse = ME.SparseTensor(features, coordinates, 
                                            coordinate_manager=output_sparse.coordinate_manager)

            loss = sparse_mse_loss(output_sparse, target_sparse)
            accuracy = reconstruction_accuracy(output_sparse, target_sparse)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_accuracy += accuracy.item()

        epoch_loss = running_loss / len(dataloader)
        recall = running_accuracy / len(dataloader)
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Recon Recall: {recall:.4f}")

        # wandb 로깅 추가
        wandb.log({
            "loss": epoch_loss,
            "reconstruction_recall": recall
        })

        if (epoch + 1) % 100 == 0:
            folder_name = f"{PROJECT_PATH}/model/3d_auto_encoder"
            os.makedirs(folder_name, exist_ok=True)
            torch.save(model.state_dict(),
                       f"{folder_name}/3d_autoencoder_{epoch+1}.pth")

if __name__ == "__main__":
    wandb.init(project='3d-auto-encoder')

    dataset = CostmapDataset()
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True,
                            collate_fn=minkowski_collate_fn)

    model = SparseVoxelAutoencoder(latent_dim=128)

    train_autoencoder(model, dataloader, epochs=10000, lr=1e-4)

    # 간단한 추론 예시
    coordinates, features = next(iter(dataloader))
    coordinates = coordinates.to(next(model.parameters()).device)
    features = features.to(next(model.parameters()).device)

    with torch.no_grad():
        reconstructed, _ = model(coordinates, features)

    print("Input Sparse shape:", features.shape)
    print("Output Sparse shape:", reconstructed.F.shape)

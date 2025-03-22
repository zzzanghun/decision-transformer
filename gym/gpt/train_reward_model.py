from datetime import datetime
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.data import Dataset, DataLoader
import pickle
import copy
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_PATH)
print(f"프로젝트 경로: {PROJECT_PATH}")

from auto_encoder.model import CostmapConvAutoencoder
from gpt.model import RewardModel

# 시드 설정
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

# 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 장치: {device}")

class TrajectoryDataset(Dataset):
    """
    궤적 데이터를 처리하는 Dataset 클래스
    """
    def __init__(self, dataset_path=None):
        """
        Parameters:
        -----------
        dataset_path : str
            데이터셋 경로
        """
        if dataset_path is None:
            dataset_path = f'{PROJECT_PATH}/data/medial/grid_4/ego-planner-data_1.pkl'
        
        self.drone_info_data = []
        self.obs_data = []
        self.path_data = []
        self.reward_data = []
        
        self._load_data(dataset_path)
    
    def _load_data(self, dataset_path):
        """
        데이터셋을 로드하고 전처리합니다.
        """
        print(f"데이터셋 로드 중: {dataset_path}")
        for i in range(1, 8):
            dataset_path = f"{PROJECT_PATH}/data/GPT_reward/gtp_reward_data_{i}.pkl"
            if i == 1:
                with open(dataset_path, 'rb') as f:
                    trajectories = pickle.load(f)
            else:
                with open(dataset_path, 'rb') as f:
                    trajectories += pickle.load(f)

        # Define the indices of the actions to be used
        action_indices = [0, 1, 2, 6, 7, 8]
        obs_indices = [0, 1, 3, 4, 6, 7, 9, 10]

        print(f"trajectories 길이: {len(trajectories)}")

        for i in range(len(trajectories)):
            episode = copy.deepcopy(trajectories[i])
            episode['actions'] = episode['actions'][:, action_indices]

            obs_first_part = episode['observations'][:, :, :100*100]
            obs_second_part = episode['observations'][:, :, 100*100:]
            obs_second_part = obs_second_part[:, :, obs_indices]
            episode['observations'] = np.concatenate([obs_first_part, obs_second_part], axis=2)

            for j in range(len(episode['actions'])):
                coef = episode['actions'][j]
                coef = np.round(coef / 0.001) * 0.001
                episode['actions'][j] = coef

                a5, a4, a3, b5, b4, b3 = coef

                drone_info_observation = []

                drone_info = episode['observations'][j][:, 100*100:]
                v_x = drone_info[0][2]         # 현재 x축 속도
                v_y = drone_info[0][3]         # 현재 y축 속도
                a_x = drone_info[0][6]         # 현재 x축 가속도
                a_y = drone_info[0][7]         # 현재 y축 가속도

                # 목표방향 정규화
                direction_vector = episode['observations'][j][:, 100*100:100*100 + 2]
                norm = np.linalg.norm(direction_vector)
                if norm != 0:
                    direction_vector = direction_vector / norm
                episode['observations'][j][:, 100*100:100*100 + 2] = direction_vector

                drone_info_observation.append(direction_vector[0][0])
                drone_info_observation.append(direction_vector[0][1])
                drone_info_observation.append(v_x)
                drone_info_observation.append(v_y)
                drone_info_observation.append(a_x)
                drone_info_observation.append(a_y)

                obs_observation = episode['observations'][j][:, :100*100].reshape(100, 100)
                path_observation = np.zeros_like(obs_observation)

                x0, y0 = 5, 5
                vx = []
                vy = []

                t_values = np.arange(0, 2.0 + 0.01, 0.01)
                for t in t_values:
                    x = x0 + v_x * t + 0.5 * a_x * t**2 + a3 * t**3 + a4 * t**4 + a5 * t**5
                    y = y0 + v_y * t + 0.5 * a_y * t**2 + b3 * t**3 + b4 * t**4 + b5 * t**5

                    v_x_t = v_x + a_x * t + 3 * a3 * t**2 + 4 * a4 * t**3 + 5 * a5 * t**4
                    v_y_t = v_y + a_y * t + 3 * b3 * t**2 + 4 * b4 * t**3 + 5 * b5 * t**4

                    ix = int(round(50 - (x - x0) * 10))
                    iy = int(round(50 - (y - y0) * 10))
                    if 0 <= ix < 100 and 0 <= iy < 100:
                        path_observation[ix, iy] = 1

                    # 0.1초 간격으로 속도 정보 저장
                    if abs((t * 10) % 1) < 1e-10:
                        drone_info_observation.append(v_x_t)
                        drone_info_observation.append(v_y_t)

                # 임의의 보상 값 생성 (실제로는 전문가 피드백이나 다른 방법으로 얻어야 함)
                # 여기서는 예시로 경로의 부드러움에 따라 보상 부여
                reward_value = trajectories[i]['rewards'][j]
                
                drone_info_observation = np.array(drone_info_observation)

                self.drone_info_data.append(drone_info_observation)
                self.obs_data.append(obs_observation)
                self.path_data.append(path_observation)
                self.reward_data.append(reward_value)
        
        print(f"데이터 로드 완료: {len(self.drone_info_data)} 샘플")

    def __len__(self):
        return len(self.drone_info_data)

    def __getitem__(self, idx):
        """
        데이터셋의 idx번째 샘플을 반환합니다.
        """
        drone_info = torch.tensor(self.drone_info_data[idx], dtype=torch.float32)
        obs = torch.tensor(self.obs_data[idx], dtype=torch.float32).unsqueeze(0)  # (1, 100, 100)
        path = torch.tensor(self.path_data[idx], dtype=torch.float32).unsqueeze(0)  # (1, 100, 100)
        reward = torch.tensor(self.reward_data[idx], dtype=torch.float32)
        
        return {
            'drone_info': drone_info,
            'obs': obs,
            'path': path,
            'reward': reward
        }


def get_dataloader(batch_size=32, shuffle=True, train_ratio=0.9):
    """
    데이터로더를 생성하여 반환합니다.
    
    Parameters:
    -----------
    batch_size : int
        배치 크기
    shuffle : bool
        데이터 셔플 여부
    train_ratio : float
        학습 데이터 비율
        
    Returns:
    --------
    tuple
        (train_dataloader, val_dataloader)
    """
    dataset = TrajectoryDataset()
    
    # 학습/검증 데이터 분할
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle
    )
    
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    
    return train_dataloader, val_dataloader


def train_reward_model(model, train_loader, val_loader, epochs=1000000, lr=1e-4):
    """
    보상 모델 학습 함수
    
    Parameters:
    -----------
    model : RewardModel
        학습할 모델
    train_loader : DataLoader
        학습 데이터로더
    val_loader : DataLoader
        검증 데이터로더
    epochs : int
        학습 에폭 수
    lr : float
        학습률
    """
    model.to(device)
    
    # 손실 함수 및 옵티마이저 설정
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # wandb 초기화
    wandb.init(project="reward-model-training", config={
        "epochs": epochs,
        "batch_size": train_loader.batch_size,
        "learning_rate": lr,
    })
    
    # 학습 기록
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # 학습 모드
        model.train()
        train_loss = 0.0
        
        # 학습 루프
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            drone_info = batch['drone_info'].to(device)
            obs = batch['obs'].to(device)
            path = batch['path'].to(device)
            target_reward = batch['reward'].to(device).unsqueeze(1)  # (B, 1)
            
            # 그래디언트 초기화
            optimizer.zero_grad()
            
            # 순전파
            predicted_reward = model(drone_info, obs, path)
            
            # 손실 계산
            loss = criterion(predicted_reward, target_reward)
            
            # 역전파 및 최적화
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * drone_info.size(0)
        
        # 에폭 평균 손실
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # 검증 모드

        if epoch % 10 == 0:
            model.eval()
            val_loss = 0.0
        
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                    drone_info = batch['drone_info'].to(device)
                    obs = batch['obs'].to(device)
                    path = batch['path'].to(device)
                    target_reward = batch['reward'].to(device).unsqueeze(1)  # (B, 1)
                    
                    # 순전파
                    predicted_reward = model(drone_info, obs, path)
                    
                    # 손실 계산
                    loss = criterion(predicted_reward, target_reward)
                    
                    val_loss += loss.item() * drone_info.size(0)
            
            # 에폭 평균 검증 손실
            val_loss /= len(val_loader.dataset)
            val_losses.append(val_loss)
            
            # 학습률 스케줄러 업데이트
            scheduler.step(val_loss)

            print(f"Epoch {epoch+1}/{epochs}, Val Loss: {val_loss:.6f}")

            wandb.log({
                "val_loss": val_loss,
                "learning_rate": optimizer.param_groups[0]['lr']
            })
        
        # 로그 출력
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}")
        
        # wandb 로깅
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
        })
        
        # 최고 성능 모델 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            folder_name = f"{PROJECT_PATH}/model/reward_model"
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
            model_save_path = f"{PROJECT_PATH}/model/reward_model/reward_model_best.pth"
            torch.save(model.state_dict(), model_save_path)
        
        # 주기적으로 모델 저장
        if (epoch + 1) % 100 == 0:
            folder_name = f"{PROJECT_PATH}/model/reward_model"
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
            model_save_path = f"{PROJECT_PATH}/model/reward_model/reward_model_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), model_save_path)
    
    # 학습 완료 후 최종 모델 저장
    model_save_path = f"{PROJECT_PATH}/model/reward_model/reward_model_final.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"최종 모델 저장됨: {model_save_path}")
    
    # wandb 종료
    wandb.finish()
    
    return train_losses, val_losses


def visualize_predictions(model, dataloader, num_samples=5):
    """
    모델 예측 시각화 함수
    
    Parameters:
    -----------
    model : RewardModel
        학습된 모델
    dataloader : DataLoader
        데이터로더
    num_samples : int
        시각화할 샘플 수
    """
    model.to(device)
    model.eval()
    
    samples = []
    for batch in dataloader:
        samples.append(batch)
        if len(samples) * dataloader.batch_size >= num_samples:
            break
    
    plt.figure(figsize=(15, num_samples * 5))
    
    sample_idx = 0
    for batch in samples:
        drone_info = batch['drone_info'].to(device)
        obs = batch['obs'].to(device)
        path = batch['path'].to(device)
        target_reward = batch['reward'].to(device)
        
        with torch.no_grad():
            predicted_reward = model(drone_info, obs, path).cpu().numpy().flatten()
        
        target_reward = target_reward.cpu().numpy()
        
        for i in range(min(dataloader.batch_size, len(drone_info))):
            if sample_idx >= num_samples:
                break
                
            plt.subplot(num_samples, 3, sample_idx * 3 + 1)
            plt.imshow(obs[i, 0].cpu().numpy(), cmap='gray')
            plt.title(f"장애물 맵 {sample_idx+1}")
            plt.axis('off')
            
            plt.subplot(num_samples, 3, sample_idx * 3 + 2)
            plt.imshow(path[i, 0].cpu().numpy(), cmap='gray')
            plt.title(f"경로 맵 {sample_idx+1}")
            plt.axis('off')
            
            plt.subplot(num_samples, 3, sample_idx * 3 + 3)
            plt.bar(['실제 보상', '예측 보상'], [target_reward[i], predicted_reward[i]])
            plt.title(f"보상 비교 {sample_idx+1}")
            plt.ylim(0, 1)
            
            sample_idx += 1
    
    plt.tight_layout()
    plt.savefig(f"{PROJECT_PATH}/gym/model/reward_model_predictions.png")
    plt.show()


if __name__ == '__main__':
    # 데이터로더 생성
    train_dataloader, val_dataloader = get_dataloader(batch_size=128)
    
    # 오토인코더 모델 로드
    obstacle_encoder = CostmapConvAutoencoder()
    path_encoder = CostmapConvAutoencoder()

    obstacle_encoder.load_state_dict(torch.load(f"{PROJECT_PATH}/model/model_900.pth"))
    path_encoder.load_state_dict(torch.load(f"{PROJECT_PATH}/model/autoencoder_with_traj_400.pth"))

    # 보상 모델 생성
    # 첫 번째 배치에서 drone_info 차원 확인
    sample_batch = next(iter(train_dataloader))
    drone_info_dim = sample_batch['drone_info'].shape[1]
    print(f"드론 정보 차원: {drone_info_dim}")
    
    reward_model = RewardModel(obstacle_encoder, path_encoder, drone_info_dim=drone_info_dim, latent_dim=128, dropout_rate=0.3)
    
    # 모델 학습
    train_losses, val_losses = train_reward_model(
        reward_model, 
        train_dataloader, 
        val_dataloader, 
        epochs=1000000, 
        lr=1e-4
    )
    
    # 학습된 모델 로드 (최고 성능 모델)
    reward_model.load_state_dict(torch.load(f"{PROJECT_PATH}/gym/model/reward_model_best.pth"))
    
    # 예측 시각화
    visualize_predictions(reward_model, val_dataloader, num_samples=5)
    
    # 샘플 데이터 확인
    sample_batch = next(iter(train_dataloader))
    print("드론 정보 형태:", sample_batch['drone_info'].shape)
    print("장애물 맵 형태:", sample_batch['obs'].shape)
    print("경로 맵 형태:", sample_batch['path'].shape)
    print("보상 형태:", sample_batch['reward'].shape)


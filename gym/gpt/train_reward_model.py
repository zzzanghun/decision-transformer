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
import math
PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_PATH)
print(f"프로젝트 경로: {PROJECT_PATH}")

from auto_encoder.model import CostmapConvAutoencoder
from gpt.model import RewardModel
from gpt.model_overfitting import RewardModelOverfitting
from gpt.model_combined import RewardModelCombined
from gpt.get_reward_from_gpt import reconstruct_from_runlength

# 시드 설정
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

# 텐서 출력 설정 - 모든 요소 표시 (수정된 버전)
torch.set_printoptions(threshold=float('inf'), linewidth=10000)  # 무한대 대신 큰 정수 사용
np.set_printoptions(threshold=np.inf, linewidth=10000)  # 무한대 대신 큰 정수 사용

# 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 장치: {device}")


def convert_to_runlength(obs_observation):
    unique_values = [val for val in np.unique(obs_observation) if val != 0]
    runlength_data = []
    
    for value in unique_values:
        positions = np.argwhere(obs_observation == value)
        if len(positions) == 0:
            continue
        
        # 값이 2인 경우 (궤적) - 직선으로 처리
        if value == 2:
            if len(positions) <= 2:
                min_row = np.min(positions[:, 0])
                max_row = np.max(positions[:, 0])
                min_col = np.min(positions[:, 1])
                max_col = np.max(positions[:, 1])
                
                runlength_data.append({
                    "rows": f"{min_row}-{max_row}",
                    "cols": f"{min_col}-{max_col}",
                    "value": "2"
                })
            else:
                # 점이 3개 이상이면 각 점을 개별적으로 처리
                for pos in positions:
                    row, col = pos
                    runlength_data.append({
                        "rows": f"{row}-{row}",
                        "cols": f"{col}-{col}",
                        "value": "2"
                    })
        else:
            # 값이 1인 경우 (장애물) - 기존 방식대로 처리
            positions = positions[np.lexsort((positions[:, 1], positions[:, 0]))]
            regions = []
            visited = set()
            
            for pos in positions:
                row, col = pos
                if (row, col) in visited:
                    continue
                    
                min_row, max_row = row, row
                min_col, max_col = col, col
                queue = [(row, col)]
                visited.add((row, col))
                
                while queue:
                    r, c = queue.pop(0)
                    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        nr, nc = r + dr, c + dc
                        if (nr, nc) not in visited and 0 <= nr < obs_observation.shape[0] and 0 <= nc < obs_observation.shape[1]:
                            if obs_observation[nr, nc] == value:
                                queue.append((nr, nc))
                                visited.add((nr, nc))
                                min_row = min(min_row, nr)
                                max_row = max(max_row, nr)
                                min_col = min(min_col, nc)
                                max_col = max(max_col, nc)
                
                regions.append({
                    "rows": f"{min_row}-{max_row}",
                    "cols": f"{min_col}-{max_col}",
                    "value": str(int(value))
                })
            
            merged_regions = []
            for region in regions:
                if not any(r["rows"] == region["rows"] and r["cols"] == region["cols"] and r["value"] == region["value"] for r in merged_regions):
                    merged_regions.append(region)
            
            runlength_data.extend(merged_regions)

    return runlength_data

class TrajectoryDataset(Dataset):
    """
    궤적 데이터를 처리하는 Dataset 클래스
    """
    def __init__(self, dataset_path=None, load_data=False):
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

        self.filtered_reward_data = 0

        if load_data:
            dataset_path = f"{PROJECT_PATH}/data/reward_model_train_data.pkl"
            with open(dataset_path, 'rb') as f:
                data = pickle.load(f)
            self.drone_info_data = data['drone_info']
            self.obs_data = data['obs']
            self.path_data = data['path']
            self.reward_data = data['reward']
        else:
            self._load_data(dataset_path)
    
    def filter_abnormal_rewards_via_velocity(self, target_direction, current_velocity, 
                                            predicted_x_velocity, predicted_y_velocity, 
                                            reward_value, min_distance):
        # 방향 벡터와 속도 벡터를 numpy 배열로 변환
        target_direction = np.array(target_direction, dtype=float)
        current_velocity = np.array(current_velocity, dtype=float)
        
        # 벡터 정규화
        target_direction_norm = np.linalg.norm(target_direction)
        current_velocity_norm = np.linalg.norm(current_velocity)
        
        # 코사인 유사도 계산 (0으로 나누기 방지)
        if target_direction_norm > 0 and current_velocity_norm > 0:
            cos_sim = np.dot(target_direction, current_velocity) / (target_direction_norm * current_velocity_norm)
        else:
            cos_sim = 0
        
        # 현재 속도 크기 계산
        velocity_magnitude = np.linalg.norm(current_velocity)
        
        # 예측 속도 처리
        if isinstance(predicted_x_velocity, (list, np.ndarray)) and isinstance(predicted_y_velocity, (list, np.ndarray)):
            if len(predicted_x_velocity) > 0 and len(predicted_y_velocity) > 0:
                # 예측 속도 벡터 생성
                predicted_velocities = np.column_stack((predicted_x_velocity, predicted_y_velocity))
                
                # 예측 속도의 평균 크기 계산
                predicted_velocity_magnitudes = np.linalg.norm(predicted_velocities, axis=1)
                mean_velocity_magnitude = np.mean(predicted_velocity_magnitudes)
                
                # 예측 속도와 타겟 방향의 정렬 정도 계산
                if target_direction_norm > 0:
                    # 각 예측 속도와 타겟 방향 사이의 코사인 유사도 계산
                    predicted_direction_alignments = []
                    for vel in predicted_velocities:
                        vel_norm = np.linalg.norm(vel)
                        if vel_norm > 0:
                            alignment = np.dot(vel, target_direction) / (vel_norm * target_direction_norm)
                            predicted_direction_alignments.append(alignment)

        mean_alignment = np.mean(predicted_direction_alignments)
        
        # 1. 코사인 유사도 기반 필터링
        if cos_sim > 0.996 and reward_value < 0.5:
            return False
        
        # 2. 속도 크기 기반 필터링
        if velocity_magnitude > 0.7 and reward_value < 0.5:
            return False
        
        if mean_alignment > 0.96 and mean_velocity_magnitude > 0.55 and min_distance <= 3 and reward_value < 0.4:
            return False
        
        if mean_alignment < 0.9 and reward_value > 0.7 and min_distance > 5:
            return False
        
        return True

    def filter_abnormal_rewards_via_obs_and_path(self, obs_matrix, path_matrix, reward_value):
        """
        장애물 행렬과 경로 행렬, 그리고 보상 값을 기반으로 비정상적인 보상을 필터링합니다.
        
        Parameters:
        -----------
        obs_matrix : numpy.ndarray
            100x100 크기의 장애물 행렬 (1은 장애물, 0은 빈 공간)
        path_matrix : numpy.ndarray
            100x100 크기의 경로 행렬 (1은 경로, 0은 빈 공간)
        reward_value : float
            GPT가 제공한 보상 값 (0~1 사이)
            
        Returns:
        --------
        bool
            True: 정상적인 보상, False: 비정상적인 보상
        float
            장애물과 경로 사이의 최소 거리
        """
        # 장애물(1)과 경로(1) 위치 찾기
        obstacle_positions = np.argwhere(obs_matrix == 1)
        path_positions = np.argwhere(path_matrix == 1)
        
        # 장애물이나 경로가 없는 경우
        if len(obstacle_positions) == 0 and (reward_value > 0.8 or reward_value < 0.2):
            return True, 100
        
        # 최소 거리 계산 (맨해튼 거리)
        min_distance = float('inf')
        for obs_pos in obstacle_positions:
            for path_pos in path_positions:
                dist = np.sqrt((obs_pos[0] - path_pos[0])**2 + (obs_pos[1] - path_pos[1])**2)
                min_distance = min(min_distance, dist)
        
        # 필터링 조건
        if min_distance <= 2 and reward_value < 0.6:
            # 장애물과 경로가 매우 가까운데 낮은 보상을 준 경우 (비정상)
            return False, min_distance
        
        if 5 < min_distance <= 10 and reward_value > 0.7:
            return False, min_distance

        return True, min_distance
    
    def _load_data(self, dataset_path):
        """
        데이터셋을 로드하고 전처리합니다.
        """
        print(f"데이터셋 로드 중: {dataset_path}")
        for i in range(1, 13):
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

        filter_reward = 0

        for i in tqdm(range(len(trajectories)), desc="에피소드 처리 중"):
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
                v_x = copy.deepcopy(drone_info[0][2])         # 현재 x축 속도
                v_y = copy.deepcopy(drone_info[0][3])         # 현재 y축 속도
                a_x = copy.deepcopy(drone_info[0][6])         # 현재 x축 가속도
                a_y = copy.deepcopy(drone_info[0][7])         # 현재 y축 가속도

                # 목표방향 정규화
                direction_vector = episode['observations'][j][:, 100*100:100*100 + 2]
                norm = np.linalg.norm(direction_vector)
                if norm != 0:
                    direction_vector = direction_vector / norm
                episode['observations'][j][:, 100*100:100*100 + 2] = copy.deepcopy(direction_vector)

                drone_info_observation.append(direction_vector[0][0])
                drone_info_observation.append(direction_vector[0][1])
                drone_info_observation.append(v_x)
                drone_info_observation.append(v_y)
                drone_info_observation.append(a_x)
                drone_info_observation.append(a_y)

                obs_observation = episode['observations'][j][:, :100*100].reshape(100, 100)
                path_observation = np.zeros_like(obs_observation)

                runlength_data = convert_to_runlength(obs_observation)
                obs_observation = reconstruct_from_runlength(runlength_data)

                x0, y0 = 5, 5
                vx = []
                vy = []

                t_values = np.arange(0, 1.7 + 0.01, 0.01)
                for t in t_values:
                    x = x0 + v_x * t + 0.5 * a_x * t**2 + a3 * t**3 + a4 * t**4 + a5 * t**5
                    y = y0 + v_y * t + 0.5 * a_y * t**2 + b3 * t**3 + b4 * t**4 + b5 * t**5

                    v_x_t = v_x + a_x * t + 3 * a3 * t**2 + 4 * a4 * t**3 + 5 * a5 * t**4
                    v_y_t = v_y + a_y * t + 3 * b3 * t**2 + 4 * b4 * t**3 + 5 * b5 * t**4

                    ix = int(round(50 - (x - x0) * 10))
                    iy = int(round(50 - (y - y0) * 10))
                    if 0 <= ix < 100 and 0 <= iy < 100:
                        path_observation[ix, iy] = 1.0

                    # 0.1초 간격으로 속도 정보 저장
                    if abs((t * 10) % 1) < 1e-10:
                        drone_info_observation.append(v_x_t)
                        drone_info_observation.append(v_y_t)
                        vx.append(v_x_t)
                        vy.append(v_y_t)

                # 임의의 보상 값 생성 (실제로는 전문가 피드백이나 다른 방법으로 얻어야 함)
                # 여기서는 예시로 경로의 부드러움에 따라 보상 부여
                reward_value = trajectories[i]['rewards'][j]
                
                drone_info_observation = np.array(drone_info_observation)

                # filter_abnormal_rewards via obs and path
                filter_obs_path, min_distance = self.filter_abnormal_rewards_via_obs_and_path(obs_observation, path_observation, reward_value)
                filter_velocity = self.filter_abnormal_rewards_via_velocity([drone_info_observation[0], drone_info_observation[1]], [drone_info_observation[2], drone_info_observation[3]], 
                                                                       vx, vy, reward_value, min_distance)
                
                if not filter_velocity:
                    self.filtered_reward_data += 1

                if filter_obs_path and filter_velocity:
                    self.drone_info_data.append(copy.deepcopy(drone_info_observation))
                    self.obs_data.append(copy.deepcopy(obs_observation))
                    self.path_data.append(copy.deepcopy(path_observation))
                    self.reward_data.append(copy.deepcopy(reward_value))

        data = {
            'drone_info': self.drone_info_data,
            'obs': self.obs_data,
            'path': self.path_data,
            'reward': self.reward_data
        }

        with open(f"{PROJECT_PATH}/data/reward_model_train_data.pkl", 'wb') as f:
            pickle.dump(data, f)
        
        print(f"데이터 저장 및 로드 완료: {len(self.drone_info_data)} 샘플, 필터링 reward 횟수: {self.filtered_reward_data}")
        # self.reward_1 = np.array(self.reward_1)
        # self.reward_2 = np.array(self.reward_2)
        # self.reward_3 = np.array(self.reward_3)
        # self.reward_4 = np.array(self.reward_4)
        # self.reward_5 = np.array(self.reward_5)
        # self.reward_6 = np.array(self.reward_6)
        
        # print(f"reward_1 mean: {np.mean(self.reward_1)}, reward_2 mean: {np.mean(self.reward_2)}, reward_3 mean: {np.mean(self.reward_3)}, reward_4 mean: {np.mean(self.reward_4)}")
        
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


def get_dataloader(batch_size=32, shuffle=True, train_ratio=0.8, load_data=False):
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
    dataset = TrajectoryDataset(load_data=load_data)
    
    # 학습/검증 데이터 분할
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    print(f"dataset 길이: {len(dataset)}")
    print(f"train_size: {train_size}, val_size: {val_size}")
    generator = torch.Generator().manual_seed(50)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=generator
    )
    
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle
    )
    
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    
    return train_dataloader, val_dataloader


def train_reward_model(model, train_loader, val_loader, epochs=1000000, lr=1e-4, l1_lambda=1e-5, use_l1_regularization=True, use_l2_regularization=True):
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
    l1_lambda : float
        L1 정규화 강도
    """
    model.to(device)
    
    # 손실 함수 및 옵티마이저 설정
    criterion = nn.MSELoss()
    if use_l2_regularization:
        optimizer = optim.Adam(model.get_trainable_parameters(), lr=lr, weight_decay=1e-4)
    else:
        optimizer = optim.Adam(model.get_trainable_parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=30, verbose=True
    )
    
    # wandb 초기화
    wandb.init(project="reward-model-training", 
               name=f"use_l1={use_l1_regularization}, use_l2={use_l2_regularization}, batch_size={train_loader.batch_size}",
               config={
                "epochs": epochs,
                "batch_size": train_loader.batch_size,
                "learning_rate": lr,
                "l1_lambda": l1_lambda,
                "use_l1_regularization": use_l1_regularization,
                "use_l2_regularization": use_l2_regularization
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
            
            # MSE 손실 계산
            mse_loss = criterion(predicted_reward, target_reward)
            
            # L1 정규화 계산
            l1_reg = 0
            for param in model.parameters():
                l1_reg += torch.sum(torch.abs(param))
            
            # 총 손실 = MSE + L1 정규화
            if use_l1_regularization:
                loss = mse_loss + (1e-6 * l1_reg)
            else:
                loss = mse_loss
            
            # 역전파 및 최적화
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * drone_info.size(0)
        
        # 에폭 평균 손실
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # 검증 모드
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

        print(f"Epoch {epoch+1}/{epochs}, Val Loss: {math.sqrt(val_loss):.6f}, Train Loss: {math.sqrt(mse_loss):.6f}")
        
        # wandb 로깅
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": math.sqrt(mse_loss),
            "l1_reg": l1_reg.item(),
            "val_loss": math.sqrt(val_loss),
            "learning_rate": optimizer.param_groups[0]['lr']
        })
        
        # 최고 성능 모델 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            folder_name = f"{PROJECT_PATH}/model/reward_model_l1_{use_l1_regularization}_l2_{use_l2_regularization}_batch_size_{train_loader.batch_size}"
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
            model_save_path = f"{PROJECT_PATH}/model/reward_model_l1_{use_l1_regularization}_l2_{use_l2_regularization}_batch_size_{train_loader.batch_size}/reward_model_best.pth"
            torch.save(model.state_dict(), model_save_path)
            print("최고 성능 모델 저장")
        
        # 주기적으로 모델 저장
        if (epoch + 1) % 100 == 0:
            folder_name = f"{PROJECT_PATH}/model/reward_model_l1_{use_l1_regularization}_l2_{use_l2_regularization}_batch_size_{train_loader.batch_size}"
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
            model_save_path = f"{PROJECT_PATH}/model/reward_model_l1_{use_l1_regularization}_l2_{use_l2_regularization}_batch_size_{train_loader.batch_size}/reward_model_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), model_save_path)
    
    # 학습 완료 후 최종 모델 저장
    folder_name = f"{PROJECT_PATH}/model/reward_model_l1_{use_l1_regularization}_l2_{use_l2_regularization}_batch_size_{train_loader.batch_size}"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    model_save_path = f"{PROJECT_PATH}/model/reward_model_l1_{use_l1_regularization}_l2_{use_l2_regularization}_batch_size_{train_loader.batch_size}/reward_model_final.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"최종 모델 저장됨: {model_save_path}")
    
    # wandb 종료
    wandb.finish()
    
    return train_losses, val_losses


if __name__ == '__main__':
    # 데이터로더 생성
    train_dataloader, val_dataloader = get_dataloader(batch_size=64, train_ratio=0.8, load_data=False)
    
    # 오토인코더 모델 로드
    obstacle_encoder = CostmapConvAutoencoder()
    path_encoder = CostmapConvAutoencoder()

    obstacle_encoder.load_state_dict(torch.load(f"{PROJECT_PATH}/model/autoencoder_with_runlength_1000.pth"))
    path_encoder.load_state_dict(torch.load(f"{PROJECT_PATH}/model/autoencoder_with_traj_400.pth"))

    # 보상 모델 생성
    # 첫 번째 배치에서 drone_info 차원 확인
    sample_batch = next(iter(train_dataloader))
    drone_info_dim = sample_batch['drone_info'].shape[1]
    print(f"드론 정보 차원: {drone_info_dim}")
    
    # reward_model = RewardModel(obstacle_encoder, path_encoder, drone_info_dim=drone_info_dim, latent_dim=128, dropout_rate=0.3)
    # reward_model = RewardModelCombined(obstacle_encoder, path_encoder, drone_info_dim=drone_info_dim, latent_dim=128, dropout_rate=0.1)
    reward_model = RewardModelOverfitting(obstacle_encoder, path_encoder, drone_info_dim=drone_info_dim, latent_dim=128)
    
    # 모델 학습
    train_losses, val_losses = train_reward_model(
        reward_model, 
        train_dataloader, 
        val_dataloader, 
        epochs=1000000, 
        lr=1e-5,
        use_l1_regularization=False,
        use_l2_regularization=True
    )
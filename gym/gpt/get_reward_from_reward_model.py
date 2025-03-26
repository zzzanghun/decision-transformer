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
import time
from tqdm import tqdm

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


def filter_abnormal_rewards_via_velocity(target_direction, current_velocity, 
                                        predicted_x_velocity, predicted_y_velocity, 
                                        reward_value):
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
    mean_velocity_magnitude = np.mean(predicted_velocity_magnitudes)

    # print(mean_alignment, ", ", mean_velocity_magnitude, ", ", velocity_magnitude, "@@@@")
    
    # 1. 코사인 유사도 기반 필터링
    if cos_sim > 0.97 and reward_value < 0.5:
        return False
    
    # 2. 속도 크기 기반 필터링
    if velocity_magnitude > 0.7 and reward_value < 0.5:
        return False
    
    if mean_alignment > 0.9 and reward_value < 0.4:
        return False
    
    if mean_alignment < 0.8 and reward_value > 0.6:
        return False
    
    if any(velocity > 0.7 for velocity in predicted_velocity_magnitudes) and reward_value < 0.5:
        return False
    
    return True

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

class RewardModelDataset():
    """
    궤적 데이터를 처리하는 Dataset 클래스
    """
    def __init__(self, reward_model=None):
        """
        Parameters:
        -----------
        dataset_path : str
            데이터셋 경로
        """
        # self.dataset_path = f'{PROJECT_PATH}/data/combined_data.pkl'
        # self.dataset_path = f'{PROJECT_PATH}/data/ego/ego-planner-data_17.pkl'
        self.dataset_path = f'{PROJECT_PATH}/data/medial/grid_4/ego-planner-data_1.pkl'
        self.reward_model = reward_model
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.reward_model.to(self.device)
        self.reward_model.eval()

    def get_reward_from_reward_model(self):
        """
        데이터셋을 로드하고 전처리합니다.
        """
        print(f"데이터셋 로드 중: {self.dataset_path}")
        with open(self.dataset_path, 'rb') as f:
            trajectories = pickle.load(f)

        # Define the indices of the actions to be used
        action_indices = [0, 1, 2, 6, 7, 8]
        obs_indices = [0, 1, 3, 4, 6, 7, 9, 10]

        print(f"trajectories 길이: {len(trajectories)}")

        # for i in range(len(trajectories)):
        for i in tqdm(range(len(trajectories)), desc="에피소드 처리 중"):
            episode = copy.deepcopy(trajectories[i])
            trajectories[i]['actions'] = trajectories[i]['actions'][:, action_indices]

            obs_first_part = trajectories[i]['observations'][:, :, :100*100]
            obs_second_part = trajectories[i]['observations'][:, :, 100*100:]
            obs_second_part = obs_second_part[:, :, obs_indices]
            trajectories[i]['observations'] = np.concatenate([obs_first_part, obs_second_part], axis=2)
            trajectories[i]['rewards'] = np.zeros(len(trajectories[i]['actions']), dtype=np.float32)

            for j in range(len(trajectories[i]['actions'])):
                coef = trajectories[i]['actions'][j].copy()
                coef = np.round(coef / 0.001) * 0.001

                a5, a4, a3, b5, b4, b3 = coef

                drone_info_observation = []

                drone_info = trajectories[i]['observations'][j][:, 100*100:]
                v_x = copy.deepcopy(drone_info[0][2])         # 현재 x축 속도
                v_y = copy.deepcopy(drone_info[0][3])         # 현재 y축 속도
                a_x = copy.deepcopy(drone_info[0][6])         # 현재 x축 가속도
                a_y = copy.deepcopy(drone_info[0][7])         # 현재 y축 가속도

                # 목표방향 정규화
                direction_vector = trajectories[i]['observations'][j][:, 100*100:100*100 + 2]
                norm = np.linalg.norm(direction_vector)
                if norm != 0:
                    direction_vector = direction_vector / norm
                trajectories[i]['observations'][j][:, 100*100:100*100 + 2] = copy.deepcopy(direction_vector)

                drone_info_observation.append(direction_vector[0][0])
                drone_info_observation.append(direction_vector[0][1])
                drone_info_observation.append(v_x)
                drone_info_observation.append(v_y)
                drone_info_observation.append(a_x)
                drone_info_observation.append(a_y)

                obs_observation = trajectories[i]['observations'][j][:, :100*100].reshape(100, 100)
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
                
                drone_info_observation = np.array(drone_info_observation)

                drone_info = torch.tensor(drone_info_observation, dtype=torch.float32).to(self.device)
                obs = torch.tensor(obs_observation, dtype=torch.float32).unsqueeze(0).to(self.device)  # (1, 100, 100)
                path = torch.tensor(path_observation, dtype=torch.float32).unsqueeze(0).to(self.device)  # (1, 100, 100)

                drone_info = drone_info.unsqueeze(0)
                obs = obs.unsqueeze(0)
                path = path.unsqueeze(0)

                with torch.no_grad():
                    reward_value = self.reward_model(drone_info, obs, path)
                reward_value = reward_value[0][0].detach().cpu().numpy()
                trajectories[i]['rewards'][j] = reward_value

                # print(obs_observation)
                filter_velocity = filter_abnormal_rewards_via_velocity([drone_info_observation[0], drone_info_observation[1]], [drone_info_observation[2], drone_info_observation[3]], 
                                                                       vx, vy, reward_value)
                # print(reward_value)

                # time.sleep(0.5)

                trajectories[i]['actions'][j] = trajectories[i]['actions'][j] / 5
                trajectories[i]['actions'][j] = np.round(trajectories[i]['actions'][j] / 0.001) * 0.001

        save_path = f"{PROJECT_PATH}/data/reward_model_combined_data.pkl"
        with open(save_path, 'wb') as f:
            pickle.dump(trajectories, f)
        print(f"데이터가 {save_path}에 저장되었습니다.")


if __name__ == '__main__':
    # 오토인코더 모델 로드
    obstacle_encoder = CostmapConvAutoencoder()
    path_encoder = CostmapConvAutoencoder()

    obstacle_encoder.load_state_dict(torch.load(f"{PROJECT_PATH}/model/autoencoder_with_runlength_1000.pth"))
    path_encoder.load_state_dict(torch.load(f"{PROJECT_PATH}/model/autoencoder_with_traj_400.pth"))

    # reward_model = RewardModelCombined(obstacle_encoder, path_encoder, drone_info_dim=48, latent_dim=128, dropout_rate=0.1)
    reward_model = RewardModelOverfitting(obstacle_encoder, path_encoder, drone_info_dim=42, latent_dim=128)
    reward_model.load_state_dict(torch.load(f"{PROJECT_PATH}/model/reward_model_best.pth"), strict=True)

    print(reward_model)

    reward_model_dataset = RewardModelDataset(reward_model=reward_model)
    reward_model_dataset.get_reward_from_reward_model()

    print("데이터 처리 완료")

import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
import time
import os

np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

class CostmapDataset(Dataset):
    """
    sampled_traj 내의 각 trajectory['observations'][..] 에서 
    100x100 costmap을 추출하여 Dataset으로 구성합니다.
    """
    def __init__(self):
        """
        Parameters:
        -----------
        sampled_traj : list
            'observations' 필드에 (배치, ..., 100*100 + 기타 정보)가 들어있는
            여러 trajectory 딕셔너리들의 리스트.
        """
        self.data = []
        
        dataset_path = f'{PROJECT_PATH}/data/combined_data.pkl'
        print(PROJECT_PATH)
        with open(dataset_path, 'rb') as f:
            trajectories = pickle.load(f)

        action_indices = [0, 1, 2, 6, 7, 8]
        obs_indices = [0, 1, 3, 4, 6, 7, 9, 10]
        
        for i in range(len(trajectories)):
            trajectories[i]['actions'] = trajectories[i]['actions'][:, action_indices]
            obs_first_part = trajectories[i]['observations'][:, :, :100*100]
            obs_second_part = trajectories[i]['observations'][:, :, 100*100:]
            obs_second_part = obs_second_part[:, :, obs_indices]
            trajectories[i]['observations'] = np.concatenate([obs_first_part, obs_second_part], axis=2)
            trajectories[i]['rewards'] = np.zeros(len(trajectories[i]['actions']), dtype=float)
            save_traj = False
            del_traj = False
            for j in range(len(trajectories[i]['actions'])):
                coef = trajectories[i]['actions'][j]
                # Discretize to 0.001 intervals
                coef = np.round(coef / 0.001) * 0.001
                # Assign back
                trajectories[i]['actions'][j] = coef
                a5, a4, a3, b5, b4, b3 = coef
                drone_info = trajectories[i]['observations'][j][:, 100*100:]
                v_x = drone_info[0][2]         # 현재 x축 속도
                v_y = drone_info[0][3]         # 현재 y축 속도
                a_x = drone_info[0][6]         # 현재 x축 가속도
                a_y = drone_info[0][7]         # 현재 y축 가속도
                direction_vector = trajectories[i]['observations'][j][:, 100*100:100*100 + 2]
                norm = np.linalg.norm(direction_vector)
                if norm != 0:
                    direction_vector = direction_vector / norm
                trajectories[i]['observations'][j][:, 100*100:100*100 + 2] = direction_vector
                obs_observation = trajectories[i]['observations'][j][:, :100*100].reshape(100, 100)
                x0, y0 = 5, 5
                t_values = np.arange(0, 1.5 + 0.01, 0.01)
                for t in t_values:
                    x = x0 + v_x * t + 0.5 * a_x * t**2 + a3 * t**3 + a4 * t**4 + a5 * t**5
                    y = y0 + v_y * t + 0.5 * a_y * t**2 + b3 * t**3 + b4 * t**4 + b5 * t**5
                    
                    # grid map의 인덱스로 변환 (여기서는 반올림하여 정수 인덱스로 변환)
                    ix = int(round(50 - (x - x0) * 10))
                    iy = int(round(50 - (y - y0) * 10))
                    
                    # grid map의 범위 내에 있는 경우에만 값을 2로 지정
                    if 0 <= ix < 100 and 0 <= iy < 100:
                        obs_observation[ix, iy] = 0.5  # 일반적으로 행이 y축, 열이 x축을 나타냄

                self.data.append(obs_observation.copy())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # (100,100)을 (1,100,100)로 확장
        costmap = self.data[idx]
        costmap = np.expand_dims(costmap, axis=0)  # (1, 100, 100)
        
        # 텐서 변환
        costmap_tensor = torch.from_numpy(costmap)
        
        return costmap_tensor

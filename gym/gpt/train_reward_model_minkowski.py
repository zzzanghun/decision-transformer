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

from gpt.model_minkowski import RewardModelMinkowski
# from gpt.get_reward_from_gpt import reconstruct_from_runlength

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


def preprocessing_obs_for_minkowski_2d(obs_observation):
    """
    2D observation을 MinkowskiEngine에서 사용할 수 있는 형식으로 전처리

    Args:
        obs_observation: (100, 100) numpy array - 2D projection된 장애물 맵

    Returns:
        coords: torch.Tensor (N, 2) - [y, x] 형식의 좌표 (batch_idx는 collate_fn에서 추가)
        feats: torch.Tensor (N, 1) - 각 좌표의 특성값
    """
    assert obs_observation.shape == (100, 100), f"Expected shape (100, 100), got {obs_observation.shape}"

    # 0이 아닌 voxel 좌표만 추출 (장애물과 궤적 모두 포함)
    coords = np.argwhere(obs_observation != 0)  # (N, 2) - [y, x]
    feats = obs_observation[obs_observation != 0].reshape(-1, 1).astype(np.float32)

    # 만약 coords에 아무런 요소가 없다면 (빈 맵인 경우)
    if coords.shape[0] == 0:
        coords = np.zeros((1, 2), dtype=np.int32)
        feats = np.zeros((1, 1), dtype=np.float32)

    coords = coords.astype(np.int32)  # (N, 2) - [y, x]

    # CPU에서 텐서로 변환 (나중에 모델에서 device로 이동)
    return torch.tensor(coords, dtype=torch.int32), torch.tensor(feats, dtype=torch.float32)

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
            dataset_path = f'/home/link/git/decision-transformer/gym/data/gpt_rtg_data_5.pkl'
        
        self.drone_info_data = []
        self.obs_data = []
        self.rtg_data = []

        self._load_data(dataset_path)
    
    def _load_data(self, dataset_path):
        """
        데이터셋을 로드하고 전처리합니다.
        """
        for i in range(1, 6):
            dataset_path = f"/home/link/git/decision-transformer/gym/data/gpt_rtg_data_{i}.pkl"
            print(f"데이터셋 로드 중: {dataset_path}")
            if i == 1:
                with open(dataset_path, 'rb') as f:
                    trajectories = pickle.load(f)
            else:
                with open(dataset_path, 'rb') as f:
                    trajectories += pickle.load(f)

        # Define the indices of the actions to be used
        action_indices = [0, 1, 2, 6, 7, 8]

        print(f"trajectories 길이: {len(trajectories)}")

        print(trajectories[0]['trajectory'].keys())

        for i in tqdm(range(len(trajectories)), desc="에피소드 처리 중"):
            episode = copy.deepcopy(trajectories[i]['trajectory'])
            episode['actions'] = episode['actions'][:, action_indices]

            for j in range(len(episode['actions'])):
                v_x = (episode['observations'][j][:, 100*100*10 + 4] * 1.5)[0]
                v_y = (episode['observations'][j][:, 100*100*10 + 5] * 1.5)[0]
                a_x = (episode['observations'][j][:, 100*100*10 + 10] * 8.0)[0]
                a_y = (episode['observations'][j][:, 100*100*10 + 11] * 8.0)[0]

                coef = episode['actions'][j]

                a5, a4, a3, b5, b4, b3 = coef

                drone_info_observation = []

                # 목표방향 정규화
                direction_vector = episode['observations'][j][:, 100*100*10:100*100*10 + 2]
                norm = np.linalg.norm(direction_vector)
                if norm != 0:
                    direction_vector = direction_vector / norm

                drone_info_observation.append(direction_vector[0][0])
                drone_info_observation.append(direction_vector[0][1])
                drone_info_observation.append(v_x)
                drone_info_observation.append(v_y)

                obs_observation = episode['observations'][j][:, :100*100*10].reshape(100, 100, 10)
                obs_observation = np.max(obs_observation, axis=2)

                # runlength_data = convert_to_runlength(obs_observation)
                # obs_observation = reconstruct_from_runlength(runlength_data)

                x0, y0 = 5, 5

                t_values = np.arange(0, 1.0 + 0.1, 0.1)
                for t in t_values:
                    x = x0 + v_x * t + 0.5 * a_x * t**2 + a3 * t**3 + a4 * t**4 + a5 * t**5
                    y = y0 + v_y * t + 0.5 * a_y * t**2 + b3 * t**3 + b4 * t**4 + b5 * t**5

                    ix = int(round(50 + (x - x0) * 10))
                    iy = int(round(50 + (y - y0) * 10))
                    if 0 <= ix < 100 and 0 <= iy < 100:
                        obs_observation[iy, ix] = -1.0  # NumPy array는 [row, col] = [y, x] 순서

                    traj_x = 50 + (x - x0) * 10
                    traj_y = 50 + (y - y0) * 10
                    drone_info_observation.append(traj_x / 50.0)
                    drone_info_observation.append(traj_y / 50.0)

                rtg_value = episode['rtg'][j]

                if int(rtg_value) not in [0, 1]:
                    continue

                drone_info_observation = np.array(drone_info_observation)

                # MinkowskiEngine용 전처리: (coords, feats) 튜플로 변환
                coords, feats = preprocessing_obs_for_minkowski_2d(obs_observation)

                self.drone_info_data.append(copy.deepcopy(drone_info_observation))
                self.obs_data.append((coords, feats))  # 전처리된 (coords, feats) 튜플 저장
                self.rtg_data.append(copy.deepcopy(rtg_value))

        # data = {
        #     'drone_info': self.drone_info_data,
        #     'obs': self.obs_data,
        #     'rtg': self.rtg_data
        # }
        
    def __len__(self):
        return len(self.drone_info_data)

    def __getitem__(self, idx):
        """
        데이터셋의 idx번째 샘플을 반환합니다.
        """
        drone_info = torch.tensor(self.drone_info_data[idx], dtype=torch.float32)
        obs = self.obs_data[idx]  # (coords, feats) 튜플 - 이미 전처리됨
        rtg = torch.tensor(self.rtg_data[idx], dtype=torch.float32)

        return {
            'drone_info': drone_info,
            'obs': obs,
            'rtg': rtg
        }


def collate_fn(batch):
    """
    DataLoader를 위한 커스텀 collate 함수
    obs가 (coords, feats) 튜플 리스트로 유지되도록 처리하며, 각 샘플에 배치 인덱스를 추가

    Parameters:
    -----------
    batch : list
        배치 내 샘플들의 리스트

    Returns:
    --------
    dict
        배치 데이터 딕셔너리
    """
    drone_info = torch.stack([item['drone_info'] for item in batch])

    # obs에 배치 인덱스 추가
    obs_with_batch_idx = []
    for batch_idx, item in enumerate(batch):
        coords, feats = item['obs']  # (N, 2), (N, 1)
        # 배치 인덱스를 coords의 첫 번째 열에 추가
        batch_idx_col = torch.full((coords.shape[0], 1), batch_idx, dtype=torch.int32)
        coords_with_batch = torch.cat([batch_idx_col, coords], dim=1)  # (N, 3) - [batch_idx, y, x]
        obs_with_batch_idx.append((coords_with_batch, feats))

    rtg = torch.stack([item['rtg'] for item in batch])

    return {
        'drone_info': drone_info,
        'obs': obs_with_batch_idx,
        'rtg': rtg
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
        train_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn
    )

    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    return train_dataloader, val_dataloader


def train_reward_model(model, train_loader, val_loader, epochs=1000000, lr=3e-4, l1_lambda=1e-5, use_l1_regularization=False, use_l2_regularization=True, warmup_epochs=10):
    """
    이진 분류 모델 학습 함수 (0: 안전, 1: 효율)

    Parameters:
    -----------
    model : RewardModelMinkowski
        학습할 모델
    train_loader : DataLoader
        학습 데이터로더
    val_loader : DataLoader
        검증 데이터로더
    epochs : int
        학습 에폭 수
    lr : float
        최대 학습률
    l1_lambda : float
        L1 정규화 강도
    warmup_epochs : int
        Warmup 에폭 수
    """
    model.to(device)

    # 손실 함수 - Binary Cross Entropy with Logits (더 안정적)
    criterion = nn.BCEWithLogitsLoss()

    # 옵티마이저 설정 - AdamW 사용 (더 나은 정규화)
    if use_l2_regularization:
        optimizer = optim.AdamW(model.get_trainable_parameters(), lr=lr, weight_decay=1e-3, betas=(0.9, 0.999))
    else:
        optimizer = optim.AdamW(model.get_trainable_parameters(), lr=lr, betas=(0.9, 0.999))

    # 학습률 스케줄러 - Cosine Annealing with Warmup
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=1e-6
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
    best_val_acc = 0.0

    for epoch in range(epochs):
        # Warmup learning rate
        if epoch < warmup_epochs:
            warmup_lr = lr * (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr

        # 학습 모드
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        # 학습 루프
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            drone_info = batch['drone_info'].to(device)
            obs = batch['obs']  # list of (coords, feats) tuples
            # coords와 feats를 device로 이동
            obs = [(coords.to(device), feats.to(device)) for coords, feats in obs]
            target_rtg = batch['rtg'].to(device).unsqueeze(1)  # (B, 1)

            assert torch.all((target_rtg == 0) | (target_rtg == 1)), f"target_rtg 값이 0 또는 1이 아님: {target_rtg}"

            # 그래디언트 초기화
            optimizer.zero_grad()

            # 순전파
            logits = model(drone_info, obs)

            # BCE with Logits 손실 계산
            bce_loss = criterion(logits, target_rtg)

            # 총 손실 = BCE + L1 정규화
            if use_l1_regularization:
                # L1 정규화 계산
                l1_reg = 0
                for param in model.parameters():
                    l1_reg += torch.sum(torch.abs(param))
                loss = bce_loss + (l1_lambda * l1_reg)
            else:
                loss = bce_loss
                l1_reg = torch.tensor(0.0)  # wandb 로깅을 위해

            # 역전파 및 최적화
            loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # 정확도 계산
            predictions = (torch.sigmoid(logits) > 0.5).float()
            train_correct += (predictions == target_rtg).sum().item()
            train_total += target_rtg.size(0)

            train_loss += loss.item() * drone_info.size(0)
        
        # 에폭 평균 손실 및 정확도
        train_loss /= len(train_loader.dataset)
        train_acc = train_correct / train_total
        train_losses.append(train_loss)

        # Learning rate scheduler step (after warmup)
        # if epoch >= warmup_epochs:
        #     scheduler.step()

        # 검증 모드
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                drone_info = batch['drone_info'].to(device)
                obs = batch['obs']  # list of (coords, feats) tuples
                # coords와 feats를 device로 이동
                obs = [(coords.to(device), feats.to(device)) for coords, feats in obs]
                target_rtg = batch['rtg'].to(device).unsqueeze(1)  # (B, 1)

                # 순전파
                logits = model(drone_info, obs)

                # 손실 계산
                loss = criterion(logits, target_rtg)

                # 정확도 계산
                predictions = (torch.sigmoid(logits) > 0.5).float()
                val_correct += (predictions == target_rtg).sum().item()
                val_total += target_rtg.size(0)

                val_loss += loss.item() * drone_info.size(0)

        # 에폭 평균 검증 손실 및 정확도
        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / val_total
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.6f}, Val Acc: {val_acc:.4f}")

        # wandb 로깅
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "l1_reg": l1_reg.item() if use_l1_regularization else 0.0,
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
    # 데이터로더 생성 - 배치 크기 증가로 안정적인 학습
    train_dataloader, val_dataloader = get_dataloader(batch_size=128, train_ratio=0.8, load_data=False)

    sample_batch = next(iter(train_dataloader))
    drone_info_dim = sample_batch['drone_info'].shape[1]
    print(f"드론 정보 차원: {drone_info_dim}")

    # 모델 생성 - latent_dim 증가로 표현력 향상
    reward_model = RewardModelMinkowski(drone_info_dim=drone_info_dim, latent_dim=256)

    # 모델 파라미터 수 출력
    total_params = sum(p.numel() for p in reward_model.parameters())
    trainable_params = sum(p.numel() for p in reward_model.parameters() if p.requires_grad)
    print(f"총 파라미터 수: {total_params:,}")
    print(f"학습 가능한 파라미터 수: {trainable_params:,}")

    # 모델 학습 - 최적화된 하이퍼파라미터
    train_losses, val_losses = train_reward_model(
        reward_model,
        train_dataloader,
        val_dataloader,
        epochs=1000000,
        lr=1e-6,  # 더 높은 초기 학습률
        l1_lambda=1e-5,
        use_l1_regularization=False,  # L2만 사용
        use_l2_regularization=True,
        warmup_epochs=500  # Warmup 추가
    )
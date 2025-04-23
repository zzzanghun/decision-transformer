import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
import os
import sys

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_PATH)

class CostmapDataset(Dataset):
    """
    MinkowskiEngine 기반 Sparse Voxel Autoencoder를 위한 데이터셋 클래스.
    voxel map 데이터 (10,50,50)을 sparse 형태로 변환하여 제공합니다.
    """
    def __init__(self, dataset_path=None):
        if dataset_path is None:
            dataset_path = f'{PROJECT_PATH}/data/3d/3d_data.pkl'

        with open(dataset_path, 'rb') as f:
            trajectories = pickle.load(f)

        self.data = []
        for traj in trajectories:
            for obs in traj['observations']:
                voxel_map = obs[:, :100*100*100].reshape(100, 100, 10)
                if np.sum(voxel_map) == 0:
                    continue
                self.data.append(voxel_map.copy())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        voxel_map = self.data[idx]  # (10,50,50)

        coords = np.argwhere(voxel_map > 0)  # 0이 아닌 voxel 좌표만 추출
        feats = voxel_map[voxel_map > 0].reshape(-1, 1).astype(np.float32)

        if coords.shape[0] == 0:
            coords = np.zeros((1, 3), dtype=np.int32)
            feats = np.zeros((1,1), dtype=np.float32)

        # batch index 추가: dataloader의 collate_fn에서 처리 예정 (일단은 0으로 넣어둠)
        batch_idx = np.zeros((coords.shape[0], 1), dtype=np.int32)
        coords = np.hstack((batch_idx, coords)).astype(np.int32)

        return {
            'coordinates': torch.tensor(coords, dtype=torch.int32),
            'features': torch.tensor(feats, dtype=torch.float32)
        }

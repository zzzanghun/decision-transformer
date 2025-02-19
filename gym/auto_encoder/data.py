import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)


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
        
        for i in range(1, 100):
            dataset_path = f'/home/zzzanghun/git/decision-transformer/gym/data/ego/odom_400/ego-planner-data_{i}.pkl'
            if i == 1:
                with open(dataset_path, 'rb') as f:
                    trajectories = pickle.load(f)
            else:
                with open(dataset_path, 'rb') as f:
                    trajectories += pickle.load(f)
        # for i in range(1, 30):
        #     dataset_path = f'/home/zzzanghun/git/decision-transformer/gym/data/medial/grid_4/ego-planner-data_{i}.pkl'
        #     with open(dataset_path, 'rb') as f:
        #         trajectories += pickle.load(f)
        # for i in range(1, 50):
        #     dataset_path = f'/home/zzzanghun/git/decision-transformer/gym/data/medial/grid_5/ego-planner-data_{i}.pkl'
        #     with open(dataset_path, 'rb') as f:
        #         trajectories += pickle.load(f)
        # for i in range(1, 100):
        #     dataset_path = f'/home/zzzanghun/git/decision-transformer/gym/data/ego/odom_300/ego-planner-data_{i}.pkl'
        #     with open(dataset_path, 'rb') as f:
        #         trajectories += pickle.load(f)
        
        for trajectory in trajectories:
            for obs in trajectory['observations']:
                # obs.shape 가 [C, 100*100 + ?] 형태라고 가정 (C=1 등)
                # 앞 100*100 부분만 costmap으로 사용
                costmap_1d = obs[:, :100*100]  # shape: (C, 10000)
                costmap_2d = costmap_1d.reshape(100, 100)  # (100, 100)
                
                # float32 변환
                costmap_2d = costmap_2d.astype(np.float32)
                
                self.data.append(costmap_2d)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # (100,100)을 (1,100,100)로 확장
        costmap = self.data[idx]
        costmap = np.expand_dims(costmap, axis=0)  # (1, 100, 100)
        
        # 텐서 변환
        costmap_tensor = torch.from_numpy(costmap)
        
        return costmap_tensor

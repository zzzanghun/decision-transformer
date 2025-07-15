import gym
from gym import spaces
import numpy as np
import random

class LineSlipEnv(gym.Env):
    """
    0 → 1 → … → N 으로 전진하는 1-D Grid-world.
    - action 0 : 왼쪽(-1), action 1 : 오른쪽(+1)
    - slip_prob 확률로 이동 방향이 반전(왼↔오른)됨.
    - 매 스텝 reward = +1  (논문·토이 실험용)
    - 목표 x == N 이면 종료, 혹은 최대 max_steps 소진 시 종료
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, N: int = 20, slip_prob: float = 0.3,
                 max_steps: int | None = None, seed: int | None = None):
        super().__init__()
        self.N = N
        self.slip_prob = slip_prob
        self.max_steps = max_steps if max_steps is not None else N
        self.action_space = spaces.Discrete(2)          # 0:L, 1:R
        self.observation_space = spaces.Discrete(N + 1) # 위치 0‥N
        self._rng = random.Random(seed)
        self.reset()

    # -------- 기본 API -------- #
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng.seed(seed)
        self.x, self.t = 0, 0
        return np.array([self.x])

    def step(self, action: int):
        # ① 의도한 이동
        move = 1 if action > 0.9 else -1
        # ② slip_prob 확률로 방향 뒤집기
        if self._rng.random() < self.slip_prob:
            move *= -1

        # ③ 실제 전이
        self.x = int(np.clip(self.x + move, 0, self.N))
        self.t += 1

        terminated  = (self.x == self.N)       # 목표 도달
        truncated   = (self.t >= self.max_steps) and not terminated
        if terminated:
            reward = 1.0
            print("terminated", self.t)
        else:
            reward = 0.0
        info        = {}
        return np.array([self.x]), reward, terminated, truncated, info

    def close(self):
        pass
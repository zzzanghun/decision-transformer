import numpy as np
import torch
from decision_transformer.envs.straight_toy_env import LineSlipEnv
import matplotlib.pyplot as plt

from decision_transformer.models.decision_transformer_toy import DecisionTransformer

def evaluate_episode(
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        device='cuda',
        target_return=None,
        mode='normal',
        state_mean=0.,
        state_std=1.,
):

    model.eval()
    model.to(device=device)
    
    env = LineSlipEnv(N=100, slip_prob=0.3, max_steps=500, seed=42)

    state = env.reset()

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, 1).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, 1), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)
    target_return = torch.tensor(target_return, device=device, dtype=torch.float32)
    sim_states = []

    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):

        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = model.get_action(
            states.to(dtype=torch.float32),
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return=target_return,
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        state, reward, terminated, truncated, _ = env.step(action)

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, 1)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        episode_return += reward
        episode_length += 1

        if terminated or truncated:
            if terminated:
                print("terminated")
            if truncated:
                print("truncated")
            break

    return episode_return, episode_length


def evaluate_episode_rtg(
        model,
        max_ep_len=1000,
        device='cuda',
        target_return=None,
        mode='noise',
    ):

    model.eval()
    model.to(device=device)
    
    env = LineSlipEnv(N=100, slip_prob=0.1, max_steps=200, seed=None)

    state = env.reset()

    state_dim = 1
    act_dim = 1

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    ep_return = target_return
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    sim_states = []
    visited_states = []  # 방문한 상태들을 저장

    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):

        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = model.get_action(
            states.to(dtype=torch.float32),
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
        )

        # print(action, "actions")

        actions[-1] = action
        action = action.detach().cpu().numpy()

        state, reward, terminated, truncated, _ = env.step(action)
        reward_noise = reward

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward_noise

        # 방문한 상태를 저장
        visited_states.append(int(state[0]))

        pred_return = target_return[0,-1] - reward
        target_return = torch.cat(
            [target_return, pred_return.reshape(1, 1)], dim=1)
        timesteps = torch.cat(
            [timesteps,
             torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)

        episode_return += reward
        episode_length += 1

        if terminated or truncated:
            if terminated:
                print("terminated")
                success = 1
                false = 0
            if truncated:
                print("truncated", cur_state)
                success = 0
                false = 1
            break

    return success, false, visited_states

if __name__ == "__main__":
    model = DecisionTransformer(
        state_dim=1,
        act_dim=1,
        max_length=10,
        max_ep_len=1000,
        hidden_size=64,
        n_layer=1,
        n_head=1,
        n_inner=4*64,
        activation_function='relu',
        n_positions=1024,
        resid_pdrop=0.0,
        attn_pdrop=0.0,
    )

    model.load_state_dict(torch.load(f"/home/link/git/decision-transformer/model/3d_end-to-end/5000_2.686392e-03/3d_model.pth"), strict=True)

    # 0~100까지의 grid 카운트 배열 초기화
    state_counts = np.zeros(101)  # 0~100까지 101개
    
    success = 0
    false = 0
    for i in range(10):
        success_i, false_i, visited_states = evaluate_episode_rtg(model, target_return=1)
        success += success_i
        false += false_i
        
        # 방문한 상태들을 카운트
        for state in visited_states:
            if 0 <= state <= 100:  # 범위 체크
                state_counts[state] += 1

    print(f"Success: {success}, False: {false}")
    print(f"Success rate: {success / (success + false)}")

    # 히트맵 그리기
    plt.figure(figsize=(12, 6))
    
    # 1D 히트맵 (바 차트)
    plt.subplot(1, 2, 1)
    plt.bar(range(101), state_counts)
    plt.xlabel('State Position')
    plt.ylabel('Visit Count')
    plt.title('State Visit Counts (Bar Chart)')
    plt.grid(True, alpha=0.3)
    
    # 2D 히트맵 (더 시각적)
    plt.subplot(1, 2, 2)
    heatmap_data = state_counts.reshape(1, -1)
    plt.imshow(heatmap_data, cmap='hot', interpolation='nearest', aspect='auto')
    plt.colorbar(label='Visit Count')
    plt.xlabel('State Position')
    plt.title('State Visit Heatmap')
    plt.yticks([])
    
    # x축 레이블 설정
    plt.xticks(range(0, 101, 10), range(0, 101, 10))
    
    plt.tight_layout()
    plt.savefig('state_visit_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 통계 정보 출력
    print(f"\n=== 방문 통계 ===")
    print(f"가장 많이 방문한 상태: {np.argmax(state_counts)} (방문 횟수: {np.max(state_counts)})")
    print(f"가장 적게 방문한 상태: {np.argmin(state_counts)} (방문 횟수: {np.min(state_counts)})")
    print(f"평균 방문 횟수: {np.mean(state_counts):.2f}")
    print(f"방문하지 않은 상태 개수: {np.sum(state_counts == 0)}")
    print(f"방문한 상태 개수: {np.sum(state_counts > 0)}")

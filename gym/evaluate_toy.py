import numpy as np
import torch
from decision_transformer.envs.straight_toy_env import LineSlipEnv
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

from decision_transformer.models.decision_transformer_toy import DecisionTransformer

def evaluate_episode(
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        device='cpu',
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
        device='cpu',
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
    total_episode_lengths = []
    episode_lengths = 0

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
            total_episode_lengths.append(t)
            break

    return success, false, visited_states, total_episode_lengths

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

    model.load_state_dict(torch.load(f"/workspace/model/3d_end-to-end/5000_8.154801e-03/only_dt_model.pth"), strict=True)

    # 0~100까지의 grid 카운트 배열 초기화
    state_counts = np.zeros(101)  # 0~100까지 101개
    
    success = 0
    false = 0
    final_episode_length = []
    iter_num = 30
    for i in range(iter_num):
        success_i, false_i, visited_states, total_episode_lengths = evaluate_episode_rtg(model, target_return=1)
        success += success_i
        false += false_i
        final_episode_length.append(total_episode_lengths[-1])
        
        # 방문한 상태들을 카운트
        for state in visited_states:
            if 0 <= state <= 100:  # 범위 체크
                state_counts[state] += 1
    state_counts[0] += iter_num

    print(f"Success: {success}, False: {false}")
    print(f"Success rate: {success / (success + false)}")
    print(f"Average episode length: {np.mean(final_episode_length)}")

    # 논문용 그래프 설정
    plt.rcParams.update({
        'font.size': 16,
        'axes.titlesize': 20,
        'axes.labelsize': 18,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 16,
        'figure.titlesize': 22
    })

    YMAX_FIXED = 200
    # 히트맵 그리기
    plt.figure(figsize=(16, 10))
    
    # gridspec을 사용하여 높이 비율 조정 (바 차트: 히트맵 = 4:1)
    gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1], hspace=0.2)
    
    # 1D 히트맵 (바 차트)
    ax1 = plt.subplot(gs[0])
    ax1.bar(range(101), state_counts, color='steelblue', alpha=0.8, edgecolor='black', linewidth=0.5)
    ax1.set_ylabel('Visit Count', fontsize=20)
    ax1.grid(True, alpha=0.3, linestyle='--')

    
    # x축 레이블 제거 (아래쪽 히트맵에만 표시)
    ax1.set_xticks(range(0, 101, 10))
    ax1.set_xticklabels(range(0, 101, 10))
    
    ax1.set_ylim(0, YMAX_FIXED)
    ax1.set_yticks(list(range(0, YMAX_FIXED + 1, 25)))
    ax1.set_yticklabels([str(t) for t in range(0, YMAX_FIXED + 1, 25)])

    # 2D 히트맵 (가로 바 형태)
    ax2 = plt.subplot(gs[1])
    heatmap_data = state_counts.reshape(1, -1)
    im = ax2.imshow(heatmap_data, cmap='coolwarm', interpolation='nearest', aspect='auto', vmin=0, vmax=YMAX_FIXED)
    
    # 컬러바를 히트맵 옆에 정확히 배치
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="2%", pad=0.1)
    cbar = plt.colorbar(im, cax=cax, label='Visit Count')
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label('Visit Count', fontsize=18)
    
    cbar.set_ticks([0, 100, YMAX_FIXED])
    cbar.ax.tick_params(labelsize=16)

    ax2.set_xlabel('State Position', fontsize=20)
    ax2.set_yticks([])
    
    # x축 레이블 설정
    ax2.set_xticks(range(0, 101, 10))
    ax2.set_xticklabels(range(0, 101, 10))
    
    # 두 축의 가로 크기를 맞춤
    pos1 = ax1.get_position()
    pos2 = ax2.get_position()
    ax2.set_position([pos1.x0, pos2.y0, pos1.width, pos2.height])
    
    plt.tight_layout()
    plt.savefig('state_visit_heatmap.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    # 통계 정보 출력
    print(f"\n=== 방문 통계 ===")
    print(f"가장 많이 방문한 상태: {np.argmax(state_counts)} (방문 횟수: {np.max(state_counts)})")
    print(f"가장 적게 방문한 상태: {np.argmin(state_counts)} (방문 횟수: {np.min(state_counts)})")
    print(f"평균 방문 횟수: {np.mean(state_counts):.2f}")
    print(f"방문하지 않은 상태 개수: {np.sum(state_counts == 0)}")
    print(f"방문한 상태 개수: {np.sum(state_counts > 0)}")

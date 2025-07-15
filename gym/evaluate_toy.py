import numpy as np
import torch
from decision_transformer.envs.straight_toy_env import LineSlipEnv

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

        print(action, "actions")

        actions[-1] = action
        action = action.detach().cpu().numpy()

        state, reward, terminated, truncated, _ = env.step(action)
        reward_noise = reward

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward_noise

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
            if truncated:
                print("truncated", cur_state)
            break

    return episode_return, episode_length

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

    episode_return, episode_length = evaluate_episode_rtg(model, target_return=1)

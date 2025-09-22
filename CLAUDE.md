# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

Create the conda environment:
```bash
conda env create -f dt.yml
conda activate dt
```

Add directories to PYTHONPATH when running scripts:
```bash
# For Atari experiments
export PYTHONPATH="${PYTHONPATH}:$(pwd)/atari"
cd atari && python run_dt_atari.py [args]

# For Gym experiments
export PYTHONPATH="${PYTHONPATH}:$(pwd)/gym"
cd gym && python experiment.py [args]
```

For gym experiments, install MuJoCo following [mujoco-py installation guide](https://github.com/openai/mujoco-py) and D4RL following [D4RL installation guide](https://github.com/rail-berkeley/d4rl).

## Common Commands

### Atari Experiments
```bash
# Decision Transformer
python run_dt_atari.py --seed 123 --context_length 30 --epochs 5 --model_type 'reward_conditioned' --num_steps 500000 --num_buffers 50 --game 'Breakout' --batch_size 128 --data_dir_prefix [DIRECTORY_NAME]

# Behavior Cloning
python run_dt_atari.py --seed 123 --context_length 30 --epochs 5 --model_type 'naive' --num_steps 500000 --num_buffers 50 --game 'Breakout' --batch_size 128 --data_dir_prefix [DIRECTORY_NAME]
```

### Gym Experiments
```bash
# Basic experiment
python experiment.py --env hopper --dataset medium --model_type dt

# With Weights & Biases logging
python experiment.py --env hopper --dataset medium --model_type dt -w True
```

### Dataset Download
```bash
# D4RL datasets for gym experiments
cd gym && python data/download_d4rl_datasets.py

# Atari DQN-replay datasets
mkdir [DIRECTORY_NAME]
gsutil -m cp -R gs://atari-replay-datasets/dqn/[GAME_NAME] [DIRECTORY_NAME]
```

## Architecture

### Two Main Domains
- **Atari**: Built on minGPT architecture, uses DQN-replay datasets
- **Gym**: Uses transformer architecture with D4RL datasets from OpenAI Gym/MuJoCo environments

### Core Implementation (`gym/decision_transformer/`)
- `models/`: Core Decision Transformer and trajectory GPT2 implementations
- `training/`: Sequence trainer and action trainer classes
- `evaluation/`: Episode evaluation utilities
- `envs/`: Custom environment implementations

### Model Types
- `reward_conditioned`: Standard Decision Transformer with return-to-go conditioning
- `naive`: Behavior cloning baseline without return conditioning
- Flow Matching VAE variants (current branch `fm_vae`)

### Training Infrastructure
- Sequence-based training for autoregressive prediction
- Action-based training for behavioral cloning
- Extensive Weights & Biases integration for experiment tracking
- CUDA support with PyTorch backend

## Development Notes

- Current development focuses on Flow Matching modules and parameter optimization
- Scripts expect to be run from their respective directories (`atari/` or `gym/`)
- All experiments log to Weights & Biases when `-w True` flag is used
- The codebase includes custom environments and evaluation metrics specific to RL benchmarks
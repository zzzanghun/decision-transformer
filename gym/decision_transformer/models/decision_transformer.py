import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import transformers
from flow_matching.path import CondOTProbPath

from decision_transformer.models.model import TrajectoryModel
from decision_transformer.models.trajectory_gpt2 import GPT2Model

import MinkowskiEngine as ME

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class DecisionTransformer(TrajectoryModel):

    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """

    def __init__(
            self,
            state_dim,
            act_dim,
            hidden_size,
            max_length=None,
            max_ep_len=4096,
            action_tanh=True,
            odom_dim=None,
            extended_cnn=False,
            time_embedding=True,
            coef_time_embedding=1,
            auto_encoder=None,
            auto_encoder_load=False,
            **kwargs
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)

        self.hidden_size = hidden_size
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            **kwargs
        )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.odom_dim = odom_dim
        self.time_embedding = time_embedding
        self.coef_time_embedding = coef_time_embedding

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = GPT2Model(config)

        self.embed_timestep = nn.Embedding(20, hidden_size)
        self.embed_return = torch.nn.Linear(1, hidden_size)

        if isinstance(self.state_dim, tuple):
            self.before_concat_hidden_size = int(hidden_size / 2)
            if auto_encoder_load:
                self.encoder = auto_encoder.encoder
                self.fc_enc = auto_encoder.fc_enc
                self.global_pool = auto_encoder.global_pool
                for param in self.encoder.parameters():
                    param.requires_grad = False
                for param in self.fc_enc.parameters():
                    param.requires_grad = False
                for param in self.global_pool.parameters():
                    param.requires_grad = False
            else:
                self.encoder = nn.Sequential(
                    ME.MinkowskiConvolution(1, 32, kernel_size=3, stride=1, dimension=3),
                    # ME.MinkowskiBatchNorm(32),
                    ME.MinkowskiReLU(inplace=True),

                    ME.MinkowskiConvolution(32, 64, kernel_size=3, stride=2, dimension=3),
                    # ME.MinkowskiBatchNorm(32),
                    ME.MinkowskiReLU(inplace=True),

                    ME.MinkowskiConvolution(64, 128, kernel_size=3, stride=2, dimension=3),
                    # ME.MinkowskiBatchNorm(64),
                    ME.MinkowskiReLU(inplace=True),

                    ME.MinkowskiConvolution(128, 256, kernel_size=3, stride=1, dimension=3),
                    # ME.MinkowskiBatchNorm(128),
                    ME.MinkowskiReLU(inplace=True),
                )
                self.drop_chlate = (ME.MinkowskiDropout(0.1))
                # Global pooling 추가
                self.global_pool = ME.MinkowskiGlobalAvgPooling()
                # latent_dim 벡터로 압축 및 복원 (dense linear 사용)
                self.norm_global_pool = nn.LayerNorm(256)
                self.drop_dense = nn.Dropout(0.05)
                self.fc_enc = nn.Linear(256, self.before_concat_hidden_size)
            self.embed_odom = nn.Sequential(
                nn.Linear(odom_dim, 2 * self.before_concat_hidden_size),
                nn.GELU(),
                nn.Linear(2 * self.before_concat_hidden_size, self.before_concat_hidden_size),
            )
            self.norm_odom = nn.LayerNorm(self.before_concat_hidden_size)
            self.norm_obstacles = nn.LayerNorm(self.before_concat_hidden_size)
        else:
            self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)

        # note: we don't predict states or returns for the paper
        self.predict_state = torch.nn.Linear(hidden_size, 1)
        self.predict_action = nn.Sequential(
            *([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
        )
        self.predict_return = torch.nn.Linear(hidden_size, 1)

        self.embed_action_time = nn.Sequential(
                nn.Linear(self.act_dim, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
        )
        self.predict_velocity = nn.Sequential(
                nn.Linear(hidden_size*2, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, self.act_dim),
        )
        self.embed_time = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
        )
        self.transformer_ln = nn.LayerNorm(hidden_size)
        self.film_gen = nn.Sequential(
                    nn.Linear(2*hidden_size, 2*hidden_size),
                    nn.GELU(),
                    nn.Linear(2*hidden_size, 2*hidden_size)
        )
        nn.init.zeros_(self.film_gen[-1].weight) 
        nn.init.zeros_(self.film_gen[-1].bias)

        self.mu = nn.Linear(hidden_size, self.act_dim)
        self.logvar = nn.Linear(hidden_size, self.act_dim)
        nn.init.constant_(self.logvar.bias, -2.0)

        self.state_ln = nn.LayerNorm(hidden_size)
        self.return_ln = nn.LayerNorm(hidden_size)
        self.path = CondOTProbPath()

        self.global_step = 0

    def forward(self, states, actions, rewards, returns_to_go, timesteps, attention_mask=None, odom=None):
        self.global_step += 1
        batch_size, seq_length = actions.shape[0], actions.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # embed each modality with a different head
        if isinstance(self.state_dim, tuple):
            # states: (Batch, Seq, 2, (coords, feats))
            # MinkowskiEngine을 사용하여 희소 복셀 처리
            
            # 시퀀스의 각 타임스텝마다 별도로 처리
            obstacles_embeddings_list = []
            
            for t in range(seq_length):
                # 현재 타임스텝의 모든 배치 데이터 수집
                coords_list, feats_list = [], []
                
                for b in range(batch_size):
                    coords = states[b][0][t][0]  # 첫 번째 [0]은 리스트 접근, 두 번째 [0]은 coords
                    feats = states[b][0][t][1]   # 첫 번째 [1]은 리스트 접근, 두 번째 [0]은 feats
                    
                    # 배치 인덱스 설정 (중요: 원본 배치 인덱스를 b로 변경)
                    coords = coords.clone()
                    coords[:, 0] = int(b)
                    coords_list.append(coords)
                    feats_list.append(feats)

                combined_coords = torch.cat(coords_list, dim=0)
                combined_feats = torch.cat(feats_list, dim=0)
                
                # MinkowskiEngine 스파스 텐서 생성
                sparse_tensor = ME.SparseTensor(
                    features=combined_feats,
                    coordinates=combined_coords,
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                )
                
                # 인코더 네트워크 통과
                x = self.encoder(sparse_tensor)
                x = self.drop_chlate(x)
                
                # 글로벌 풀링으로 각 배치 항목을 고정 크기 벡터로 변환
                x = self.global_pool(x)
                x = self.norm_global_pool(x.F)
                x = F.gelu(x)     
                x = self.drop_dense(x)
                # 최종 임베딩 생성
                embeddings = self.fc_enc(x)

                assert embeddings.shape[0] == batch_size
                
                obstacles_embeddings_list.append(embeddings)
            
            # 시퀀스 차원을 따라 임베딩 스택
            obstacles_embeddings = torch.stack(obstacles_embeddings_list, dim=1)
            
            # odom 임베딩과 결합
            odom_embeddings = self.embed_odom(odom)

            odom_embeddings = self.norm_odom(odom_embeddings)
            obstacles_embeddings = self.norm_obstacles(obstacles_embeddings)

            state_embeddings = torch.cat((obstacles_embeddings, odom_embeddings), dim=-1)
        else:
            state_embeddings = self.embed_state(states)
        
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        if self.time_embedding:
            time_embeddings = self.embed_timestep(timesteps)
            time_embeddings *= self.coef_time_embedding

            # time embeddings are treated similar to positional embeddings
            state_embeddings = state_embeddings + time_embeddings
            action_embeddings = action_embeddings + time_embeddings
            returns_embeddings = returns_embeddings + time_embeddings

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 3*seq_length)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)

        # get predictions
        # return_preds = self.predict_return(x[:,2])  # predict next return given state and action
        # state_preds = self.predict_state(x[:,2])    # predict next state given state and action
        # action_preds = self.predict_action(x[:,1])  # predict next action given state

        # print(x[:,1].shape, "@@@@@@")

        # flat actions, x[:,1]
        actions_flat = actions.reshape(-1, self.act_dim)[attention_mask.reshape(-1) > 0]
        h_flat = x[:,1].reshape(-1, self.hidden_size)[attention_mask.reshape(-1) > 0]
        h_flat = self.transformer_ln(h_flat)

        # create t, x_t, u_t
        t = torch.rand(actions_flat.shape[0]).to(self.device)

        x0, mu, logvar = self.x0_reparameterize(h_flat)

        path_sample = self.path.sample(t=t, x_0=x0, x_1=actions_flat)
        x_t = path_sample.x_t
        u_t = path_sample.dx_t.detach()

        # t embedding
        t = timestep_embedding(t, self.hidden_size)
        t = self.embed_time(t)

        # h_flat + t
        h_flat_t = torch.cat([h_flat, t], dim=-1)

        # Film Gen
        gamma_beta = self.film_gen(h_flat_t)
        gamma, beta = gamma_beta.chunk(2, dim=-1)

        # x_t embedding
        x_t = self.embed_action_time(x_t)

        # adapt Film to x_t
        x_t = x_t * (1 + gamma) + beta

        state_embeddings = state_embeddings.reshape(-1, self.hidden_size)[attention_mask.reshape(-1) > 0]
        state_embeddings = self.state_ln(state_embeddings)
        state_embeddings = self.drop_dense(state_embeddings)

        x_t = torch.cat([x_t, state_embeddings], dim=-1)

        # predict u_t
        u_t_pred = self.predict_velocity(x_t)

        return u_t, u_t_pred, mu, logvar

    def x0_reparameterize(self, h):
        mu = self.mu(h)
        logvar = self.logvar(h).clamp(-5.0, 5.0)
        std = torch.exp(0.5 * logvar)

        # --- optional: epsilon clipping for early stability ---
        eps = torch.randn_like(std)
        eps = eps.clamp_(-2.5, 2.5)

        # --- noise annealing α(schedule on std*eps only) ---
        step = self.global_step
        warmup_steps = 50000
        alpha_min = 0.01

        p = float(step) / float(warmup_steps)
        alpha = 0.5 - 0.5 * math.cos(math.pi * p)
        alpha = alpha_min + (1.0 - alpha_min) * alpha

        z = mu + (alpha * std) * eps
        return z, mu, logvar

    def get_action(self, states, actions, rewards, returns_to_go, timesteps, **kwargs):
        # we don't care about the past rewards in this model

        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if self.max_length is not None:
            states = states[:,-self.max_length:]
            actions = actions[:,-self.max_length:]
            returns_to_go = returns_to_go[:,-self.max_length:]
            timesteps = timesteps[:,-self.max_length:]

            # pad all tokens to sequence length
            attention_mask = torch.cat([torch.zeros(self.max_length-states.shape[1]), torch.ones(states.shape[1])])
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)
            states = torch.cat(
                [torch.zeros((states.shape[0], self.max_length-states.shape[1], self.state_dim), device=states.device), states],
                dim=1).to(dtype=torch.float32)
            actions = torch.cat(
                [torch.zeros((actions.shape[0], self.max_length - actions.shape[1], self.act_dim),
                             device=actions.device), actions],
                dim=1).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                [torch.zeros((returns_to_go.shape[0], self.max_length-returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go],
                dim=1).to(dtype=torch.float32)
            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], self.max_length-timesteps.shape[1]), device=timesteps.device), timesteps],
                dim=1
            ).to(dtype=torch.long)
        else:
            attention_mask = None

        _, action_preds, return_preds = self.forward(
            states, actions, None, returns_to_go, timesteps, attention_mask=attention_mask, **kwargs)

        return action_preds[0,-1]

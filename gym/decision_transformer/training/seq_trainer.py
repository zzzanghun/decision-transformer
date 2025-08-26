import numpy as np
import torch
import torch.nn.functional as F

from decision_transformer.training.trainer import Trainer


class SequenceTrainer(Trainer):

    def train_step(self, iter_num):
        self.train_num += 1
        states, actions, rewards, dones, rtg, timesteps, attention_mask, odom = self.get_batch(self.batch_size)
        action_target = torch.clone(actions)

        u_t, u_t_pred, mu = self.model.forward(
            states, actions, rewards, rtg[:,:-1], timesteps, attention_mask=attention_mask, odom=odom
        )

        # act_dim = actions.shape[2]
        # action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        # action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        # action_target_for_prev = action_target[:-1, :]
        # action_preds_for_prev = action_preds[1:, :]

        loss = self.loss_fn(
            None, u_t_pred, None,
            None, u_t, None,
        )

        prior_loss = self.prior_match_simple(mu)

        # # 더 안전한 적응형 계수 계산
        # with torch.no_grad():
        #     loss_magnitude = loss.item()
        #     prior_loss_magnitude = prior_loss.item()
            
        #     if prior_loss_magnitude > 1e-6:  # 더 안전한 임계값
        #         ratio = loss_magnitude / prior_loss_magnitude
        #         target_ratio = 20.0
        #         adaptive_coef = (ratio / target_ratio).clamp(0.001, 10.0)  # 범위 제한
        #     else:
        #         adaptive_coef = 0.01

        loss = loss + 0.01 * prior_loss

        # loss_for_prev_pred = self.loss_fn(
        #     None, action_preds_for_prev, None,
        #     None, action_target_for_prev, None
        # )

        # loss = loss_for_current_pred + 0.5 * loss_for_prev_pred

        self.optimizer.zero_grad()
        loss.backward()
        if iter_num > 50:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 3.0)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
        self.optimizer.step()

        # with torch.no_grad():
            # self.diagnostics['training/action_error'] = torch.mean((action_preds-action_target)**2).detach().cpu().item()
            # action_preds[:, :] = action_preds[:, :]
            # action_target[:, :] = action_target[:, :]
            # self.diagnostics['training/action_error'] = torch.mean((action_preds-action_target)**2).detach().cpu().item()

        return loss.detach().cpu().item(), grad_norm.detach().cpu().item(), prior_loss.detach().cpu().item()

    def _var(self, x, dim=0): # 안정적 분산 
        return x.var(dim=dim, unbiased=False).clamp_min(1e-8)

    def prior_match_simple(self, mu, s=0.4): 
        mean_loss = (mu.mean(dim=0).pow(2).mean()) 
        var_loss = (self._var(mu, 0) - s**2).pow(2).mean() 
        return mean_loss + var_loss
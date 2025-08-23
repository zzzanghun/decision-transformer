import numpy as np
import torch
import torch.nn.functional as F

from decision_transformer.training.trainer import Trainer


class SequenceTrainer(Trainer):

    def train_step(self, iter_num):
        self.train_num += 1
        states, actions, rewards, dones, rtg, timesteps, attention_mask, odom = self.get_batch(self.batch_size)
        action_target = torch.clone(actions)

        u_t, u_t_pred, mu, logvar = self.model.forward(
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

        kl = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1.0, dim=-1)
        kl_loss = kl.mean()

        beta_target = 0.01
        warmup_steps = 10000
        beta_kl = min(beta_target, beta_target * (self.train_num + 1) / warmup_steps)

        loss = loss + beta_kl * kl_loss

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

        return loss.detach().cpu().item(), grad_norm.detach().cpu().item(), kl_loss.detach().cpu().item(), mu.mean().detach().cpu().item(), logvar.mean().detach().cpu().item()

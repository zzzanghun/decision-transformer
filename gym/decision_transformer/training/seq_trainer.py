import numpy as np
import torch

from decision_transformer.training.trainer import Trainer


class SequenceTrainer(Trainer):

    def train_step(self):
        self.train_num += 1
        states, actions, rewards, dones, rtg, timesteps, attention_mask, odom = self.get_batch(self.batch_size)
        action_target = torch.clone(actions)

        state_preds, action_preds, reward_preds = self.model.forward(
            states, actions, rewards, rtg[:,:-1], timesteps, attention_mask=attention_mask, odom=odom
        )

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        # action_target_for_prev = action_target[:-1, :]
        # action_preds_for_prev = action_preds[1:, :]

        # print(action_target.shape, action_preds.shape)

        loss = self.loss_fn(
            None, action_preds, None,
            None, action_target, None,
        )

        # loss_for_prev_pred = self.loss_fn(
        #     None, action_preds_for_prev, None,
        #     None, action_target_for_prev, None
        # )

        # loss = loss_for_current_pred + 0.5 * loss_for_prev_pred

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        with torch.no_grad():
            # self.diagnostics['training/action_error'] = torch.mean((action_preds-action_target)**2).detach().cpu().item()
            action_preds[:, :] = action_preds[:, :] * 5.0
            action_target[:, :] = action_target[:, :] * 5.0
            self.diagnostics['training/action_error'] = torch.mean((action_preds-action_target)**2).detach().cpu().item()

        return loss.detach().cpu().item()

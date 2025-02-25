"""
Modified MAPPO Implementation with Value Decomposition using a Mixing Network (similar to QMIX)
and KL Penalty in PPO updates.

Changes compared to the original MAPPO algorithm:
1. Integration of a value mixing network that decomposes the global value function by combining
   individual agent value estimates using hypernetworks. This is inspired by the QMIX approach.
2. Replacement of the standard PPO clipping mechanism with a KL penalty approach for the policy update.
3. Use of TD(λ) for computing advantages and returns.
4. Centralized critic that incorporates state, observation, and agent identity information.
"""

import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from network.ppo_net import PPOActor
from network.ppo_net import PPOCritic
from torch.distributions import Categorical

# Mixing network for value decomposition (similar to QMIX)
class ValueMixingNetwork(nn.Module):
    def __init__(self, n_agents, state_dim, mixing_hidden_dim):
        super(ValueMixingNetwork, self).__init__()
        self.n_agents = n_agents
        self.mixing_hidden_dim = mixing_hidden_dim

        # Hypernetwork to generate weights for the first layer: ensures non-negative weights
        self.hyper_w_1 = nn.Sequential(
            nn.Linear(state_dim, mixing_hidden_dim * n_agents),
            nn.ReLU()
        )
        # Hypernetwork to generate weights for the second layer
        self.hyper_w_2 = nn.Sequential(
            nn.Linear(state_dim, mixing_hidden_dim),
            nn.ReLU()
        )
        self.hyper_b_1 = nn.Linear(state_dim, mixing_hidden_dim)
        self.hyper_b_2 = nn.Linear(state_dim, 1)

    def forward(self, agent_values, state):
        """
        agent_values: shape (batch, n_agents)
        state: shape (batch, state_dim)
        Returns: global value (batch,)
        """
        batch_size = agent_values.size(0)
        # First layer
        w1 = torch.abs(self.hyper_w_1(state)).view(batch_size, self.n_agents, self.mixing_hidden_dim)
        b1 = self.hyper_b_1(state).view(batch_size, 1, self.mixing_hidden_dim)
        hidden = torch.bmm(agent_values.unsqueeze(1), w1) + b1  # (batch, 1, mixing_hidden_dim)
        hidden = F.relu(hidden)
        # Second layer
        w2 = torch.abs(self.hyper_w_2(state)).view(batch_size, self.mixing_hidden_dim, 1)
        b2 = self.hyper_b_2(state).view(batch_size, 1, 1)
        y = torch.bmm(hidden, w2) + b2  # (batch, 1, 1)
        return y.view(batch_size)

class MAPPO:
    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        actor_input_shape = self.obs_shape
        critic_input_shape = self._get_critic_input_shape()

        if args.last_action:
            actor_input_shape += self.n_actions
        if args.reuse_network:
            actor_input_shape += self.n_agents
        self.args = args

        self.policy_rnn = PPOActor(actor_input_shape, args)
        self.eval_critic = PPOCritic(critic_input_shape, self.args)
        # Initialize value mixer to combine individual agent values into a global value
        self.value_mixer = ValueMixingNetwork(self.n_agents, self.state_shape, args.mixing_hidden_dim)

        if self.args.use_gpu:
            self.policy_rnn.cuda()
            self.eval_critic.cuda()
            self.value_mixer.cuda()

        self.model_dir = os.path.join(args.model_dir, args.alg, args.map)

        self.ac_parameters = list(self.policy_rnn.parameters()) + \
                             list(self.eval_critic.parameters()) + \
                             list(self.value_mixer.parameters())

        if args.optimizer == "RMS":
            self.ac_optimizer = torch.optim.RMSprop(self.ac_parameters, lr=args.lr)
        elif args.optimizer == "Adam":
            self.ac_optimizer = torch.optim.Adam(self.ac_parameters, lr=args.lr)

        self.policy_hidden = None
        self.eval_critic_hidden = None

    def _get_critic_input_shape(self):
        # Start with the state dimension
        input_shape = self.state_shape
        # Add observation dimension
        input_shape += self.obs_shape
        # Add agent identity information
        input_shape += self.n_agents
        return input_shape

    def learn(self, batch, max_episode_len, train_step, time_steps=0):
        episode_num = batch['o'].shape[0]
        self.init_hidden(episode_num)
        for key in batch.keys():
            if key == 'u':
                batch[key] = torch.tensor(batch[key], dtype=torch.long)
            else:
                batch[key] = torch.tensor(batch[key], dtype=torch.float32)
        u, r, avail_u, terminated, s = batch['u'], batch['r'], batch['avail_u'], batch['terminated'], batch['s']

        mask = (1 - batch["padded"].float())

        if self.args.use_gpu:
            u = u.cuda()
            mask = mask.cuda()
            r = r.cuda()
            terminated = terminated.cuda()
            s = s.cuda()

        # Assume initial mask shape is (episode, time)
        # Repeat along agent dimension -> (episode, time, n_agents)
        mask = mask.repeat(1, 1, self.n_agents)
        r = r.repeat(1, 1, self.n_agents)
        terminated = terminated.repeat(1, 1, self.n_agents)

        # Get old critic values and policy action probabilities (for KL penalty computation)
        old_values, _ = self._get_values(batch, max_episode_len)
        old_values = old_values.squeeze(dim=-1)
        old_action_prob = self._get_action_prob(batch, max_episode_len).detach()
        old_dist = Categorical(old_action_prob)
        old_log_pi_taken = old_dist.log_prob(u.squeeze(dim=-1))
        old_log_pi_taken[mask == 0] = 0.0

        loss_list = []

        # Compute advantages and returns using TD(λ)
        values, _ = self._get_values(batch, max_episode_len)
        values = values.squeeze(dim=-1)

        returns = torch.zeros_like(r)
        deltas = torch.zeros_like(r)
        advantages = torch.zeros_like(r)

        prev_return = 0.0
        prev_value = 0.0
        prev_advantage = 0.0
        for transition_idx in reversed(range(max_episode_len)):
            returns[:, transition_idx] = r[:, transition_idx] + self.args.gamma * prev_return * (1 - terminated[:, transition_idx]) * mask[:, transition_idx]
            deltas[:, transition_idx] = r[:, transition_idx] + self.args.gamma * prev_value * (1 - terminated[:, transition_idx]) * mask[:, transition_idx] - values[:, transition_idx]
            advantages[:, transition_idx] = deltas[:, transition_idx] + self.args.gamma * self.args.lamda * prev_advantage * (1 - terminated[:, transition_idx]) * mask[:, transition_idx]

            prev_return = returns[:, transition_idx]
            prev_value = values[:, transition_idx]
            prev_advantage = advantages[:, transition_idx]

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = advantages.detach()

        if self.args.use_gpu:
            advantages = advantages.cuda()

        # Perform updates over several PPO epochs
        for _ in range(self.args.ppo_n_epochs):
            self.init_hidden(episode_num)

            # Compute agent-level critic values and combine them into a global value via the mixing network
            values, _ = self._get_values(batch, max_episode_len)
            values = values.squeeze(dim=-1)

            # Compute returns computed earlier (returns: shape (episode, time, n_agents))
            # Since the reward is common, take the mean as the global target
            returns_global = returns.mean(dim=2)

            # Compute global value at each timestep using the mixing network
            global_values = []
            for t in range(max_episode_len):
                # Get individual agent values at timestep t: (episode, n_agents)
                agent_vals = values[:, t, :]
                # Get the corresponding state: (episode, state_dim)
                state_t = s[:, t, :]
                global_val = self.value_mixer(agent_vals, state_t)
                global_values.append(global_val)
            global_values = torch.stack(global_values, dim=1)  # (episode, time)

            # Use a single agent's mask since the mask is the same for all agents
            mask_single = mask[:, :, 0]

            # Compute value loss using MSE (without clipping)
            value_loss = 0.5 * (((global_values - returns_global) ** 2) * mask_single).sum() / mask_single.sum()

            # Compute new action probabilities from the policy
            action_prob = self._get_action_prob(batch, max_episode_len)
            dist = Categorical(action_prob)
            new_log_pi_taken = dist.log_prob(u.squeeze(dim=-1))
            new_log_pi_taken[mask == 0] = 0.0
            ratios = torch.exp(new_log_pi_taken - old_log_pi_taken.detach())

            # Use KL penalty instead of clipping (PPO with KL penalty)
            pg_loss = - (ratios * advantages * mask).sum() / mask.sum()
            kl_div = torch.distributions.kl_divergence(old_dist, dist)
            kl_div[mask == 0] = 0.0
            kl_loss = self.args.kl_coeff * (kl_div * mask).sum() / mask.sum()

            entropy = dist.entropy()
            entropy[mask == 0] = 0.0

            # Entropy bonus to encourage exploration (subtracted since we minimize loss)
            policy_loss = pg_loss + kl_loss - self.args.entropy_coeff * (entropy * mask).sum() / mask.sum()

            loss = policy_loss + value_loss
            loss_list.append(loss.item())

            self.ac_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.ac_parameters, self.args.grad_norm_clip)
            self.ac_optimizer.step()

        avg_loss = sum(loss_list) / len(loss_list)
        print("Training Step {}: Average Loss = {:.6f}".format(train_step, avg_loss))
        return avg_loss

    def _get_critic_inputs(self, batch, transition_idx, max_episode_len):
        obs, obs_next, s, s_next = batch['o'][:, transition_idx], batch['o_next'][:, transition_idx],\
                                   batch['s'][:, transition_idx], batch['s_next'][:, transition_idx]
        s = s.unsqueeze(1).expand(-1, self.n_agents, -1)
        s_next = s_next.unsqueeze(1).expand(-1, self.n_agents, -1)
        episode_num = obs.shape[0]
        inputs, inputs_next = [], []
        inputs.append(s)
        inputs_next.append(s_next)
        inputs.append(obs)
        inputs_next.append(batch['o_next'][:, transition_idx])
        inputs.append(torch.eye(self.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
        inputs_next.append(torch.eye(self.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
        inputs = torch.cat([x.reshape(episode_num * self.n_agents, -1) for x in inputs], dim=1)
        inputs_next = torch.cat([x.reshape(episode_num * self.n_agents, -1) for x in inputs_next], dim=1)
        return inputs, inputs_next

    def _get_values(self, batch, max_episode_len):
        episode_num = batch['o'].shape[0]
        v_evals = []
        for transition_idx in range(max_episode_len):
            inputs, _ = self._get_critic_inputs(batch, transition_idx, max_episode_len)
            if self.args.use_gpu:
                inputs = inputs.cuda()
                self.eval_critic_hidden = self.eval_critic_hidden.cuda()
            v_eval, self.eval_critic_hidden = self.eval_critic(inputs, self.eval_critic_hidden)
            v_eval = v_eval.view(episode_num, self.n_agents, -1)
            v_evals.append(v_eval)
        v_evals = torch.stack(v_evals, dim=1)
        return v_evals, None

    def _get_actor_inputs(self, batch, transition_idx):
        obs = batch['o'][:, transition_idx]
        u_onehot = batch['u_onehot'][:]
        episode_num = obs.shape[0]
        inputs = [obs]
        if self.args.last_action:
            if transition_idx == 0:
                inputs.append(torch.zeros_like(u_onehot[:, transition_idx]))
            else:
                inputs.append(u_onehot[:, transition_idx - 1])
        if self.args.reuse_network:
            inputs.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
        inputs = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _get_action_prob(self, batch, max_episode_len):
        episode_num = batch['o'].shape[0]
        avail_actions = batch['avail_u']
        action_prob = []
        for transition_idx in range(max_episode_len):
            inputs = self._get_actor_inputs(batch, transition_idx)
            if self.args.use_gpu:
                inputs = inputs.cuda()
                self.policy_hidden = self.policy_hidden.cuda()
            outputs, self.policy_hidden = self.policy_rnn(inputs, self.policy_hidden)
            outputs = outputs.view(episode_num, self.n_agents, -1)
            prob = F.softmax(outputs, dim=-1)
            action_prob.append(prob)
        action_prob = torch.stack(action_prob, dim=1).cpu()
        action_prob = action_prob + 1e-10
        action_prob[avail_actions == 0] = 0.0
        action_prob = action_prob / action_prob.sum(dim=-1, keepdim=True)
        action_prob[avail_actions == 0] = 0.0
        action_prob = action_prob + 1e-10
        if self.args.use_gpu:
            action_prob = action_prob.cuda()
        return action_prob

    def init_hidden(self, episode_num):
        self.policy_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))
        self.eval_critic_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))

    def save_model(self, train_step):
        num = str(train_step // self.args.save_cycle)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        torch.save(self.eval_critic.state_dict(), os.path.join(self.model_dir, num + '_critic_params.pkl'))
        torch.save(self.policy_rnn.state_dict(), os.path.join(self.model_dir, num + '_rnn_params.pkl'))

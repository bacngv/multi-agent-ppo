"""Modified MAPPO, designed in a value decomposition style. 
This implementation improves the value loss component of the original MAPPO by calculating it using value decomposition that aggregates values through a mixing network. 
This network is inspired by the concept of PPO. In the mixing network, I use the function y = log(1 + alpha * x) to stabilize the algorithm—this function behaves like a parabola so that when x is large, 
the mixed value does not increase sharply.
"""
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from network.ppo_net import PPOActor, PPOCritic
from torch.distributions import Categorical

# Mixing network for value decomposition (similar to QMIX) - updated version
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
        # Layer 1:
        w1 = torch.abs(self.hyper_w_1(state)).view(batch_size, self.n_agents, self.mixing_hidden_dim)
        b1 = self.hyper_b_1(state).view(batch_size, 1, self.mixing_hidden_dim)
        pre_activation = torch.bmm(agent_values.unsqueeze(1), w1) + b1  # (batch, 1, mixing_hidden_dim)
        
        # Apply the activation function log(1 + alpha * x)
        # Ensure non-negative input:
        activated = pre_activation.clamp(min=0)
        alpha = 2.5  # The alpha parameter can be adjusted as needed
        hidden = torch.log(1 + alpha * activated)
        
        # Layer 2:
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
        # Initialize the mixing network for value decomposition
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
        # state + obs + one-hot agent id
        input_shape = self.state_shape + self.obs_shape + self.n_agents
        return input_shape

    def learn(self, batch, max_episode_len, train_step, time_steps=0):
        episode_num = batch['o'].shape[0]
        self.init_hidden(episode_num)
        # Convert numpy arrays to tensors
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

        # Expand mask, reward, and terminated to include agents
        mask = mask.repeat(1, 1, self.n_agents)
        r = r.repeat(1, 1, self.n_agents)
        terminated = terminated.repeat(1, 1, self.n_agents)

        # Compute old action probabilities and detach to avoid gradient flow
        old_action_prob = self._get_action_prob(batch, max_episode_len).detach()
        old_dist = Categorical(old_action_prob)
        old_log_pi_taken = old_dist.log_prob(u.squeeze(dim=-1))
        old_log_pi_taken[mask == 0] = 0.0

        loss_list = []

        # Compute critic values and advantages
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

        # PPO update epochs
        for _ in range(self.args.ppo_n_epochs):
            self.init_hidden(episode_num)
            # Recompute critic values
            values, _ = self._get_values(batch, max_episode_len)
            values = values.squeeze(dim=-1)

            # Compute global returns using the mixing network (instead of a simple average)
            global_returns = []
            for t in range(max_episode_len):
                agent_returns = returns[:, t, :]  # (episode, n_agents)
                state_t = s[:, t, :]              # (episode, state_dim)
                global_return = self.value_mixer(agent_returns, state_t)
                global_returns.append(global_return)
            global_returns = torch.stack(global_returns, dim=1)  # (episode, time)

            # Compute global values using the mixing network from the critic values
            global_values = []
            for t in range(max_episode_len):
                agent_vals = values[:, t, :]  # (episode, n_agents)
                state_t = s[:, t, :]          # (episode, state_dim)
                global_val = self.value_mixer(agent_vals, state_t)
                global_values.append(global_val)
            global_values = torch.stack(global_values, dim=1)  # (episode, time)

            mask_single = mask[:, :, 0]
            value_loss = 0.5 * (((global_values - global_returns) ** 2) * mask_single).sum() / mask_single.sum()

            # Policy update: compute new action probabilities
            action_prob = self._get_action_prob(batch, max_episode_len)
            dist = Categorical(action_prob)
            log_pi_taken = dist.log_prob(u.squeeze(dim=-1))
            log_pi_taken[mask == 0] = 0.0
            ratios = torch.exp(log_pi_taken - old_log_pi_taken.detach())

            # Clipped PPO objective
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1.0 - self.args.clip_param, 1.0 + self.args.clip_param) * advantages
            policy_loss = - (torch.min(surr1, surr2) * mask).sum() / mask.sum()

            # Entropy bonus
            entropy = dist.entropy()
            entropy[mask == 0] = 0.0
            policy_loss = policy_loss - self.args.entropy_coeff * (entropy * mask).sum() / mask.sum()

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
        obs = batch['o'][:, transition_idx]
        obs_next = batch['o_next'][:, transition_idx]
        s = batch['s'][:, transition_idx]
        s_next = batch['s_next'][:, transition_idx]
        s = s.unsqueeze(1).expand(-1, self.n_agents, -1)
        s_next = s_next.unsqueeze(1).expand(-1, self.n_agents, -1)
        episode_num = obs.shape[0]
        inputs, inputs_next = [], []
        inputs.append(s)
        inputs_next.append(s_next)
        inputs.append(obs)
        inputs_next.append(obs_next)
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

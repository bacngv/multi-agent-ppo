import numpy as np
import os
from common.rollout import RolloutWorker
from agent.agent import Agents
import matplotlib.pyplot as plt
from smac.env import StarCraft2Env
import pandas as pd

class Runner:
    def __init__(self, env, args):
        self.env = env
        env = StarCraft2Env(map_name=args.map,
                            step_mul=args.step_mul,
                            difficulty=args.difficulty,
                            game_version=args.game_version,
                            replay_dir=args.replay_dir)
        self.agents = Agents(args)
        self.rolloutWorker = RolloutWorker(env, self.agents, args)
        self.args = args

        self.save_path = self.args.result_dir + '/' + args.alg + '/' + args.map
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def run(self, num=0):
        # Reset log data cho lần chạy hiện tại
        self.win_rates = []
        self.episode_rewards = []
        self.eval_timesteps = []
        self.loss_history = []
        self.train_steps_history = []

        time_steps, train_steps, evaluate_steps = 0, 0, -1
        while time_steps < self.args.n_steps:
            print('Run {}, time_steps {}'.format(num, time_steps))
            
            # Nếu đã đến chu kỳ evaluation, thực hiện đánh giá và vẽ đồ thị
            if time_steps // self.args.evaluate_cycle > evaluate_steps:
                win_rate, episode_reward = self.evaluate()
                self.win_rates.append(win_rate)
                self.episode_rewards.append(episode_reward)
                self.eval_timesteps.append(time_steps)
                self.plt(num)         # Vẽ đồ thị win rate & episode rewards
                evaluate_steps += 1

            episodes = []
            for episode_idx in range(self.args.n_episodes):
                episode, _, _, steps = self.rolloutWorker.generate_episode(episode_idx)
                episodes.append(episode)
                time_steps += steps

            episode_batch = episodes[0]
            episodes.pop(0)
            for episode in episodes:
                for key in episode_batch.keys():
                    episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)

            # Hàm train của Agents trả về loss trung bình của vòng train hiện tại
            loss = self.agents.train(episode_batch, train_steps, time_steps)
            self.loss_history.append(loss)
            self.train_steps_history.append(train_steps)  # Lưu lại bước train hiện tại
            train_steps += self.args.ppo_n_epochs
            self.plot_loss(num)

        # Sau khi train xong, thực hiện đánh giá và vẽ đồ thị cuối cùng
        win_rate, episode_reward = self.evaluate()
        print('win_rate is ', win_rate)
        self.win_rates.append(win_rate)
        self.episode_rewards.append(episode_reward)
        self.eval_timesteps.append(time_steps)
        self.plt(num)
        self.plot_loss(num)


    def evaluate(self):
        win_number = 0
        episode_rewards = 0
        for epoch in range(self.args.evaluate_epoch):
            _, episode_reward, win_tag, _ = self.rolloutWorker.generate_episode(epoch, evaluate=True)
            episode_rewards += episode_reward
            if win_tag:
                win_number += 1
        return win_number / self.args.evaluate_epoch, episode_rewards / self.args.evaluate_epoch

    def plt(self, num):
        x = self.eval_timesteps

        plt.figure(figsize=(12, 8))

        # --- Subplot 1: Win Rate ---
        plt.subplot(2, 1, 1)
        plt.plot(x, self.win_rates, label='Win Rate')
        plt.xlabel('Timesteps')
        plt.ylabel('Win Rate')
        plt.ylim(0, 1.0)
        plt.legend()

        # --- Subplot 2: Episode Rewards ---
        plt.subplot(2, 1, 2)
        plt.plot(x, self.episode_rewards, label='Episode Rewards')
        plt.xlabel('Timesteps')
        plt.ylabel('Episode Rewards')
        plt.legend()

        plt.tight_layout()

        # Save the figure
        plt.savefig(self.save_path + f'/plt_{num}.png', format='png')
        # Save the data
        data = pd.DataFrame({
            'time_steps': self.eval_timesteps,
            'win_rate': self.win_rates,
            'episode_rewards': self.episode_rewards
        })
        data.to_csv(self.save_path + f'/results_{num}.csv', index=False)
        plt.close()
        
    def plot_loss(self, num):
        plt.figure(figsize=(12, 8))
        plt.plot(self.train_steps_history, self.loss_history, label='Loss', color='red')
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.title('Loss Over Training Steps')
        plt.legend()
        plt.tight_layout()

        # Save the loss figure
        loss_plot_path = self.save_path + f'/loss_plot_{num}.png'
        plt.savefig(loss_plot_path, format='png')
        plt.close()

        # Save the loss data to a CSV file
        loss_csv_path = self.save_path + f'/loss_data_{num}.csv'
        loss_df = pd.DataFrame({
            'training_steps': self.train_steps_history,
            'loss': self.loss_history
        })
        loss_df.to_csv(loss_csv_path, index=False)

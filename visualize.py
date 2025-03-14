import os
import glob
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_results(algo_dir):
    """
    Read files results_*.csv from the algorithm's directory,
    interpolate values based on a common set of time_steps,
    and return: time_steps, win_rate_mean, win_rate_std, reward_mean, reward_std.
    """
    results_files = sorted(glob.glob(os.path.join(algo_dir, "results_*.csv")))
    if not results_files:
        print("No results_*.csv file found in directory:", algo_dir)
        return None, None, None, None, None

    experiments = []
    union_steps = set()
    for file in results_files:
        data = pd.read_csv(file)
        x = data['time_steps'].values
        win_rate = data['win_rate'].values
        reward = data['episode_rewards'].values
        union_steps.update(x.tolist())
        experiments.append((x, win_rate, reward))
    union_steps = np.array(sorted(union_steps))
    
    # Interpolate values for each training run
    win_rate_interp = []
    reward_interp = []
    for (x, wr, rw) in experiments:
        wr_interp = np.interp(union_steps, x, wr)
        rw_interp = np.interp(union_steps, x, rw)
        win_rate_interp.append(wr_interp)
        reward_interp.append(rw_interp)
    win_rate_interp = np.array(win_rate_interp)
    reward_interp = np.array(reward_interp)
    
    win_rate_mean = np.mean(win_rate_interp, axis=0)
    win_rate_std = np.std(win_rate_interp, axis=0)
    reward_mean = np.mean(reward_interp, axis=0)
    reward_std = np.std(reward_interp, axis=0)
    
    return union_steps, win_rate_mean, win_rate_std, reward_mean, reward_std

def load_loss(algo_dir):
    """
    Read files loss_data_*.csv from the algorithm's directory,
    interpolate loss values based on a common set of training_steps,
    and return: training_steps, loss_mean, loss_std.
    """
    loss_files = sorted(glob.glob(os.path.join(algo_dir, "loss_data_*.csv")))
    if not loss_files:
        print("No loss_data_*.csv file found in directory:", algo_dir)
        return None, None, None

    experiments = []
    union_steps = set()
    for file in loss_files:
        data = pd.read_csv(file)
        x = data['training_steps'].values
        loss = data['loss'].values
        union_steps.update(x.tolist())
        experiments.append((x, loss))
    union_steps = np.array(sorted(union_steps))
    
    loss_interp = []
    for (x, loss) in experiments:
        loss_i = np.interp(union_steps, x, loss)
        loss_interp.append(loss_i)
    loss_interp = np.array(loss_interp)
    
    loss_mean = np.mean(loss_interp, axis=0)
    loss_std = np.std(loss_interp, axis=0)
    return union_steps, loss_mean, loss_std

def compare_results(algo_dirs, algo_names, save_folder):
    """
    Compare the results (Win Rate and Episode Rewards) of multiple algorithms.
    Plots two subplots:
      - Subplot 1: Win Rate comparison.
      - Subplot 2: Episode Rewards comparison.
    Each algorithm is plotted with its mean line along with the ± std region.
    """
    plt.figure(figsize=(12, 12))
    
    # Subplot 1: Win Rate
    ax1 = plt.subplot(2, 1, 1)
    for algo_dir, name in zip(algo_dirs, algo_names):
        x, win_rate_mean, win_rate_std, reward_mean, reward_std = load_results(algo_dir)
        if x is None:
            continue
        ax1.plot(x, win_rate_mean, label=f"{name} Win Rate")
        ax1.fill_between(x, win_rate_mean - win_rate_std, win_rate_mean + win_rate_std, alpha=0.2)
    ax1.set_xlabel("Time Steps")
    ax1.set_ylabel("Win Rate")
    ax1.set_ylim(0, 1)
    ax1.legend()
    
    # Subplot 2: Episode Rewards
    ax2 = plt.subplot(2, 1, 2)
    for algo_dir, name in zip(algo_dirs, algo_names):
        x, win_rate_mean, win_rate_std, reward_mean, reward_std = load_results(algo_dir)
        if x is None:
            continue
        ax2.plot(x, reward_mean, label=f"{name} Episode Rewards")
        ax2.fill_between(x, reward_mean - reward_std, reward_mean + reward_std, alpha=0.2)
    ax2.set_xlabel("Time Steps")
    ax2.set_ylabel("Episode Rewards")
    ax2.legend()
    
    plt.tight_layout()
    save_path = os.path.join(save_folder, "results_comparison.png")
    plt.savefig(save_path, format='png')
    plt.close()
    print(f"Comparison results chart has been saved at {save_path}")

def compare_loss(algo_dirs, algo_names, save_folder):
    """
    Compare the loss values of multiple algorithms.
    Plot a chart with the mean loss line and ± std region for each algorithm.
    """
    plt.figure(figsize=(12, 6))
    for algo_dir, name in zip(algo_dirs, algo_names):
        x, loss_mean, loss_std = load_loss(algo_dir)
        if x is None:
            continue
        plt.plot(x, loss_mean, label=f"{name} Loss")
        plt.fill_between(x, loss_mean - loss_std, loss_mean + loss_std, alpha=0.2)
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title("Loss Comparison Across Algorithms")
    plt.legend()
    plt.tight_layout()
    
    save_path = os.path.join(save_folder, "loss_comparison.png")
    plt.savefig(save_path, format='png')
    plt.close()
    print(f"Loss comparison chart has been saved at {save_path}")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize and compare performance of multiple algorithms."
    )
    parser.add_argument("--base_dir", type=str, default="/content/multi-agent-ppo/results/",
                        help="Base directory containing the results of the algorithms.")
    parser.add_argument("--algos", type=str, nargs="+", default=["ippo", "mappo"],
                        help="List of algorithm names to compare.")
    parser.add_argument("--map", type=str, default="3m", 
                        help="Subfolder name to append after the algorithm name (e.g., 3m).")
    parser.add_argument("--save_folder", type=str, default=None,
                        help="Directory to save the result charts. Default is base_dir.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    base_dir = args.base_dir
    algo_names = args.algos
    map_folder = args.map
    algo_dirs = [os.path.join(base_dir, algo, map_folder) for algo in algo_names]
    save_folder = args.save_folder if args.save_folder else base_dir
    compare_results(algo_dirs, algo_names, save_folder)
    compare_loss(algo_dirs, algo_names, save_folder)

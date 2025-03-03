import os
import matplotlib.pyplot as plt
import numpy as np


def plot_learning_curve(episodes, records, title, ylabel, figure_file):
    plt.figure()
    plt.plot(episodes, records, linestyle='-', color='r')
    plt.title(title)
    plt.xlabel('episode')
    plt.ylabel(ylabel)

    plt.show()
    plt.savefig(figure_file)


def create_directory(path: str, sub_dirs: list):
    for sub_dir in sub_dirs:
        if os.path.exists(path + sub_dir):
            print(path + sub_dir + ' is already exist!')
        else:
            os.makedirs(path + sub_dir, exist_ok=True)
            print(path + sub_dir + ' create successfully!')


def plot_validate_curve(episodes, baseline, records, records2, title, ylabel, figure_file):
    plt.figure()
    plt.plot(episodes, records, linestyle=':', linewidth = 2, color='r', label='ppo')
    plt.plot(episodes, records2, linestyle='--', linewidth = 2, color='y', label='ddqn')
    plt.plot(episodes, baseline, linestyle='-', linewidth = 2, color='c', label='baseline')
    plt.legend()
    plt.title(title)
    plt.xlabel('episode')
    plt.ylabel(ylabel)

    plt.show()
    plt.savefig(figure_file)


def plot_validate_performance_curve(episodes, record1, record2, title, ylabel, figure_file):
    plt.figure()
    plt.plot(episodes, record1, linestyle=':', linewidth = 2, color='r', label='ddqn')
    plt.plot(episodes, record2, linestyle='--', linewidth = 2, color='y', label='ppo')
    plt.legend()
    plt.title(title)
    plt.xlabel('time')
    plt.ylabel(ylabel)

    plt.show()
    plt.savefig(figure_file)


def moving_average(interval, windowsize):
    window = np.ones(int(windowsize)) / float(windowsize)
    re = np.convolve(interval, window, 'same')
    return re


def LabberRing(windowsize, episodes, baseline, ddqn, ppo, title, ylabel, figure_file):
    ddqn_av = moving_average(ddqn, windowsize)
    ppo_av = moving_average(ppo, windowsize)
    ddqn_mean = [np.mean(ddqn) for _ in range(len(episodes))]
    ppo_mean = [np.mean(ppo) for _ in range(len(episodes))]
    plt.figure()
    plt.plot(episodes, ppo, linestyle='-', linewidth = 2, color='lightcoral', label='ppo', alpha=0.2)
    plt.plot(episodes, ppo_av, linestyle=':', linewidth = 2, color='r', label='ppo-smooth')
    plt.plot(episodes, ddqn, linestyle='-', linewidth = 2, color='skyblue', label='ddqn', alpha=0.2)
    plt.plot(episodes, ddqn_av, linestyle='--', linewidth = 2, color='b', label='ddqn-smooth')
    plt.plot(episodes, baseline, linestyle='-', linewidth = 2, color='c', label='baseline')
    plt.plot(episodes, ddqn_mean, linestyle='--', linewidth = 2, color='k')
    plt.plot(episodes, ppo_mean, linestyle=':', linewidth = 2, color='k')
    plt.text(4900, np.mean(ddqn)-1, "ddqn_convergence(+16.84%)")
    plt.text(5000, np.mean(ppo)+1, "ppo_convergence(+27.76%)")
    plt.legend(loc='lower right')
    plt.title(title)
    plt.xlabel('episode')
    plt.ylabel(ylabel)
 
    # plt.grid()网格线设置
    # plt.grid(True)
    plt.show()
    plt.savefig(figure_file)


# @Theodore Wolf

import numpy as np
import random
from IPython.display import clear_output
import torch
# import shap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Tuple
from collections import Counter
from torch import Tensor

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ReplayBuffer:
    """To store experience for uncorrelated learning"""

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class PER_IS_ReplayBuffer:
    """
    Adapted from https://github.com/labmlai/annotated_deep_learning_paper_implementations
    """

    def __init__(self, capacity, alpha, state_dim=3):
        self.capacity = capacity
        self.alpha = alpha
        self.priority_sum = [0 for _ in range(2 * self.capacity)]
        self.priority_min = [float('inf') for _ in range(2 * self.capacity)]
        self.max_priority = 1.
        self.data = {
            'obs': np.zeros(shape=(capacity, state_dim), dtype=np.float64),
            'action': np.zeros(shape=capacity, dtype=np.int32),
            'reward': np.zeros(shape=capacity, dtype=np.float32),
            'next_obs': np.zeros(shape=(capacity, state_dim), dtype=np.float64),
            'done': np.zeros(shape=capacity, dtype=bool)
        }
        self.next_idx = 0
        self.size = 0

    def push(self, obs, action, reward, next_obs, done):
        idx = self.next_idx
        self.data['obs'][idx] = obs
        self.data['action'][idx] = action
        self.data['reward'][idx] = reward
        self.data['next_obs'][idx] = next_obs
        self.data['done'][idx] = done

        self.next_idx = (idx + 1) % self.capacity
        self.size = min(self.capacity, self.size + 1)

        priority_alpha = self.max_priority ** self.alpha
        self._set_priority_min(idx, priority_alpha)
        self._set_priority_sum(idx, priority_alpha)

    def _set_priority_min(self, idx, priority_alpha):
        idx += self.capacity
        self.priority_min[idx] = priority_alpha
        while idx >= 2:
            idx //= 2
            self.priority_min[idx] = min(self.priority_min[2 * idx], self.priority_min[2 * idx + 1])

    def _set_priority_sum(self, idx, priority_alpha):
        idx += self.capacity
        self.priority_sum[idx] = priority_alpha
        while idx >= 2:
            idx //= 2
            self.priority_sum[idx] = self.priority_sum[2 * idx] + self.priority_sum[2 * idx + 1]

    def _sum(self):
        return self.priority_sum[1]

    def _min(self):
        return self.priority_min[1]

    def find_prefix_sum_idx(self, prefix_sum):
        idx = 1
        while idx < self.capacity:
            if self.priority_sum[idx * 2] > prefix_sum:
                idx = 2 * idx
            else:
                prefix_sum -= self.priority_sum[idx * 2]
                idx = 2 * idx + 1

        return idx - self.capacity

    def sample(self, batch_size, beta):

        samples = {
            'weights': np.zeros(shape=batch_size, dtype=np.float32),
            'indexes': np.zeros(shape=batch_size, dtype=np.int32),
        }

        for i in range(batch_size):
            p = random.random() * self._sum()
            idx = self.find_prefix_sum_idx(p)
            samples['indexes'][i] = idx

        prob_min = self._min() / self._sum()
        max_weight = (prob_min * self.size) ** (-beta)

        for i in range(batch_size):
            idx = samples['indexes'][i]
            prob = self.priority_sum[idx + self.capacity] / self._sum()
            weight = (prob * self.size) ** (-beta)
            samples['weights'][i] = weight / max_weight

        for k, v in self.data.items():
            samples[k] = v[samples['indexes']]

        return samples

    def update_priorities(self, indexes, priorities):

        for idx, priority in zip(indexes, priorities):
            self.max_priority = max(self.max_priority, priority)
            priority_alpha = priority ** self.alpha
            self._set_priority_min(idx, priority_alpha)
            self._set_priority_sum(idx, priority_alpha)

    def is_full(self):
        return self.capacity == self.size

    def __len__(self):
        return self.size


class MADDPG_ReplayBuffer:
    def __init__(self, capacity, obs_dims, batch_size: int):

        self.capacity = int(capacity)
        self.entries = 0

        self.batch_size = batch_size

        self.obs_dims = obs_dims
        self.max_obs_dim = np.max(obs_dims)
        self.n_agents = len(obs_dims)

        self.memory_obs = []
        self.memory_nobs = []
        for ii in range(self.n_agents):
            self.memory_obs.append(Tensor(self.capacity, obs_dims[ii]))
            self.memory_nobs.append(Tensor(self.capacity, obs_dims[ii]))
        self.memory_acts = Tensor(self.n_agents, self.capacity)
        self.memory_rwds = Tensor(self.n_agents, self.capacity)
        self.memory_dones = Tensor(self.n_agents, self.capacity)

    def store(self, obs, acts, rwds, nobs, dones):
        store_index = self.entries % self.capacity

        for ii in range(self.n_agents):
            self.memory_obs[ii][store_index] = Tensor(obs[ii])
            self.memory_nobs[ii][store_index] = Tensor(nobs[ii])
        self.memory_acts[:, store_index] = Tensor(acts)
        self.memory_rwds[:, store_index] = Tensor(rwds)
        self.memory_dones[:, store_index] = Tensor(dones)

        self.entries += 1

    def sample(self):
        if not self.ready(): return None

        idxs = np.random.choice(
            np.min((self.entries, self.capacity)),
            size=(self.batch_size,),
            replace=False,
        )

        return {
            "obs": [self.memory_obs[ii][idxs] for ii in range(self.n_agents)],
            "acts": self.memory_acts[:, idxs],
            "rwds": self.memory_rwds[:, idxs],
            "nobs": [self.memory_nobs[ii][idxs] for ii in range(self.n_agents)],
            "dones": self.memory_dones[:, idxs],
        }

    def ready(self):
        return (self.batch_size <= self.entries)


class RunningMeanStd(object):
    """
    Taken from: https://github.com/semitable/fast-marl
    """
    def __init__(self, epsilon: float = 1e-4, shape: Tuple[int, ...] = (), device="cpu"):
        """
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        """
        self.mean = torch.zeros(shape, dtype=torch.float32, device=device)
        self.var = torch.ones(shape, dtype=torch.float32, device=device)
        self.count = epsilon

    def update(self, arr):
        #arr = arr.reshape(-1, arr.size(-1))
        batch_mean = torch.mean(arr, dim=1)
        batch_var = torch.var(arr, dim=1)
        batch_count = arr.shape[1]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count: int):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = (
            m_a
            + m_b
            + torch.square(delta)
            * self.count
            * batch_count
            / (self.count + batch_count)
        )
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count


def plot(data_dict):
    """For tracking experiment progress"""
    rewards = data_dict['moving_avg_rewards']
    std = data_dict['moving_std_rewards']
    frame_idx = data_dict['frame_idx']
    clear_output(True)
    plt.figure(figsize=(20, 5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    reward = np.array(rewards)
    stds = np.array(std)
    plt.fill_between(np.arange(len(reward)), reward - 0.25 * stds, reward + 0.25 * stds, color='b', alpha=0.1)
    plt.fill_between(np.arange(len(reward)), reward - 0.5 * stds, reward + 0.5 * stds, color='b', alpha=0.1)
    plt.show()


def plot_test_trajectory(env, agent, fig, axes, max_steps=600, test_state=None, fname=None,):
    """To plot trajectories of the agent"""
    state = env.reset_for_state(test_state)
    learning_progress = []
    for step in range(max_steps):
        list_state = env.get_plot_state_list()

        # take recommended action
        action = agent.get_action(state, testing=True)

        # Do the new chosen action in Environment
        new_state, reward, done, _ = env.step(action)

        learning_progress.append([list_state, action, reward])

        state = new_state
        if done:
            break

    fig, axes = env.plot_run(learning_progress, fig=fig, axes=axes, fname=fname, )

    return fig, axes

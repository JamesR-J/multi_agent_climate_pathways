import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# from envs.AYS_Environment_MultiAgent import *


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def numpy_to_cuda(numpy_array):
    return torch.from_numpy(numpy_array).float().to(DEVICE)


class NN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.out = nn.Linear(hidden_dim // 2, action_dim)  # TODO should this be // 2 or nah

    def forward(self, x):
        x = F.relu(self.fc1(x.float()))  # TODO traceback why need a float here
        x = F.relu(self.fc2(x))
        q_val = self.out(x)
        return q_val


class DuellingNN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(DuellingNN, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.a = nn.Linear(hidden_dim // 2, action_dim)  # TODO same here inni
        self.v = nn.Linear(hidden_dim // 2, 1)  # TODO same here inni

    def forward(self, x):
        x = F.relu(self.fc1(x.float()))  # TODO traceback why need a float here
        x = F.relu(self.fc2(x))
        advantages = self.a(x)
        value = self.v(x)
        a_values = value + (advantages - advantages.mean())

        return a_values


class DQN:
    def __init__(self, state_dim, action_dim, gamma=0.99, lr=0.002357, tau=0.0877, rho=0.7052, epsilon=1., polyak=False,
                 decay=0.5, step_decay=50000):
        self.target_net = NN(state_dim, action_dim).to(DEVICE)
        self.policy_net = NN(state_dim, action_dim).to(DEVICE)

        self.lr = lr
        self.decay = decay
        self.step_decay = step_decay
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, gamma=self.decay, step_size=step_decay)
        self.action_size = action_dim
        self.gamma = gamma

        self.loss = nn.MSELoss()

        self.tau = tau
        self.counter = 0
        self.polyak = polyak

        self.t = 1
        self.rho = rho
        self.epsilon = lambda t: 0.01 + epsilon / (t ** self.rho)

    @torch.no_grad()
    def get_action(self, state: np.ndarray, testing=False):
        self.t += 1
        if np.random.uniform() > self.epsilon(self.t) or testing:
            q_values = self.policy_net(torch.Tensor(state).to(DEVICE)).cpu().numpy()
            # print(q_values)
            # print(np.argmax(q_values))
            # sys.exit()
            return np.argmax(q_values)
        else:
            return np.random.choice(self.action_size)

    def update(self, batch_sample, weights=None):
        """To update our networks"""
        # Unpack batch: 5-tuple
        state, action, reward, next_state, done = batch_sample

        # convert to torch.cuda
        states = numpy_to_cuda(state)
        actions = numpy_to_cuda(action).type(torch.int64).unsqueeze(1)
        next_states = numpy_to_cuda(next_state)
        rewards = numpy_to_cuda(reward)

        # get the Q-values of the actions at time t
        state_qs = self.policy_net(states).gather(1, actions).squeeze(1)

        # get the max Q-values at t+1 from the target network
        next_state_values = self.next_state_value_estimation(next_states, done)

        # target: y_t = r_t + gamma * max[Q(s,a)]
        if rewards.shape == (256, 1):  # TODO 256 is batch size and hardcoded also in the NN
            targets = (rewards.squeeze(1) + self.gamma * next_state_values.squeeze(1))
        else:
            targets = (rewards + self.gamma * next_state_values.squeeze(1))

        # if we have weights from importance sampling
        if weights is not None:
            weights = numpy_to_cuda(weights)
            loss = ((targets - state_qs).pow(2) * weights).mean()
        # otherwise we use the standard MSE loss
        else:
            loss = self.loss(state_qs, targets)

        # back propagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # scheduler step
        self.scheduler.step()

        # to copy the policy parameters to the target network
        self.copy_nets()
        # we return the loss for logging and the TDs for Prioritised Experience Replay
        return loss, (state_qs - targets).detach()

    @torch.no_grad()
    def next_state_value_estimation(self, next_states, done):
        """Function to define the value of the next state, makes inheritance cleaner"""
        next_state_values = self.target_net(next_states).max(1)[0].detach()
        # the value of a state after a terminal state is 0
        next_state_values[done.squeeze(1)] = 0
        return next_state_values.unsqueeze(1)

    def copy_nets(self):
        """Copies the parameters from the policy network to the target network, either all at once or incrementally."""
        self.counter += 1
        if not self.polyak and self.counter >= 1 / self.tau:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.counter = 0
        else:
            for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

    def __str__(self):
        return "DQN"


class DuelDDQN(DQN):
    def __init__(self, state_dim, action_dim, lr=0.004133, rho=0.5307, tau=0.01856, **kwargs):
        super(DuelDDQN, self).__init__(state_dim, action_dim, lr=lr, rho=rho, tau=tau, **kwargs)

        self.target_net = DuellingNN(state_dim, action_dim).to(DEVICE)
        self.policy_net = DuellingNN(state_dim, action_dim).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, gamma=self.decay, step_size=self.step_decay)

    @torch.no_grad()
    def next_state_value_estimation(self, next_states, done):
        """next state value estimation is different for DDQN,
        decouples action selection and evaluation for reduced estimation bias"""
        # find max valued action with policy net
        max_actions = self.policy_net(next_states).argmax(1).unsqueeze(1)
        # estimate value of best action with target net
        next_state_values = self.target_net(next_states).gather(1, max_actions)
        next_state_values[done] = 0
        return next_state_values

    def __str__(self):
        return "DuelDDQN"
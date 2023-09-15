import argparse
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from learn.gradient_estimators import *
from learn.agent import Agent
from typing import List, Optional
from learn.utils import RunningMeanStd
import torch.optim as optim

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


class abstract_agent(nn.Module):
    def __init__(self):
        super(abstract_agent, self).__init__()

    def act(self, input):
        policy, value = self.forward(input)  # flow the input through the nn
        return policy, value


class actor_agent(abstract_agent):
    def __init__(self, num_inputs, action_size, args):
        super(actor_agent, self).__init__()
        self.linear_a1 = nn.Linear(num_inputs, args.num_units_1)
        self.linear_a2 = nn.Linear(args.num_units_1, args.num_units_2)
        self.linear_a = nn.Linear(args.num_units_2, action_size)

        self.linear_a1_bn = nn.LayerNorm(num_inputs)
        self.linear_a2_bn = nn.LayerNorm(args.num_units_1)

        self.reset_parameters()

        self.LReLU = nn.LeakyReLU(0.01)
        self.tanh = nn.Tanh()
        self.train()

        self.nn_multiplier = 0

    def reset_parameters(self):
        gain = nn.init.calculate_gain('leaky_relu')
        gain_tanh = nn.init.calculate_gain('tanh')
        self.linear_a1.weight.data.mul_(gain)
        self.linear_a2.weight.data.mul_(gain)
        self.linear_a.weight.data.mul_(gain_tanh)

    def forward(self, input, model_original_out=False):
        """
        The forward func defines how the data flows through the graph(layers)
        """
        x = F.relu(self.linear_a1(self.linear_a1_bn(input)))
        x = F.relu(self.linear_a2(self.linear_a2_bn(x)))

        policy = self.linear_a(x)  # * max_action_size # needed if actions are clamped at the output

        policy_adj = policy  # * self.nn_multiplier

        if model_original_out:
            return policy_adj, policy_adj

        return policy_adj


class critic_agent(abstract_agent):
    def __init__(self, obs_shape_n, action_shape_n, args):
        super(critic_agent, self).__init__()
        self.linear_o_c1 = nn.Linear(obs_shape_n, args.num_units_1)
        self.linear_a_c1 = nn.Linear(action_shape_n, args.num_units_1)
        self.linear_c2 = nn.Linear(args.num_units_1 * 2, args.num_units_2)
        self.linear_c = nn.Linear(args.num_units_2, 1)

        self.linear_o_c1_bn = nn.LayerNorm(obs_shape_n)
        self.linear_a_c1_bn = nn.LayerNorm(action_shape_n)
        self.linear_c2_bn = nn.LayerNorm(args.num_units_1 * 2)

        self.reset_parameters()

        self.LReLU = nn.LeakyReLU(0.01)
        self.tanh = nn.Tanh()
        self.train()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('leaky_relu')
        gain_tanh = nn.init.calculate_gain('tanh')
        self.linear_o_c1.weight.data.mul_(gain)
        self.linear_a_c1.weight.data.mul_(gain)
        self.linear_c2.weight.data.mul_(gain)
        self.linear_c.weight.data.mul_(gain)

    def forward(self, obs_input, action_input, model_original_out=False):
        x_o = F.relu(self.linear_o_c1(self.linear_o_c1_bn(obs_input)))
        x_a = F.relu(self.linear_a_c1(self.linear_a_c1_bn(action_input)))

        x_cat = torch.cat([x_o, x_a], dim=1)

        x = F.relu(self.linear_c2(self.linear_c2_bn(x_cat)))

        value = self.linear_c(x)

        if model_original_out:
            return value, value

        return value


class DQN:
    def __init__(self, state_dim, action_dim, gamma=0.99, lr=0.002357, tau=0.0877, rho=0.7052, epsilon=1., polyak=False,
                 decay=0.5, step_decay=50000, rational_choice="2nd_best"):
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

        self.rational_choice = rational_choice

    @torch.no_grad()
    def get_action(self, state: np.ndarray, testing=False, rational=True):
        self.t += 1
        if np.random.uniform() > self.epsilon(self.t) or testing:
            q_values = self.policy_net(torch.Tensor(state).to(DEVICE)).cpu().numpy()

            if rational:
                return np.argmax(q_values)
            else:
                max_ind = np.argmax(q_values)
                q_dict = dict(enumerate(q_values))
                q_dict.pop(max_ind)
                if self.rational_choice == "2nd_best":
                    return list(q_dict.keys())[list(q_dict.values()).index(max(list(q_dict.values())))]
                elif self.rational_choice == "random":
                    return np.random.choice(list(q_dict.keys()))
                else:
                    print("INCORRECT RATIONAL CHOICE")
                    sys.exit()

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


class MADDPG:
    def __init__(
        self,
        env,
        critic_lr : float,
        actor_lr : float,
        gradient_clip : float,
        hidden_dim_width : int,
        gamma : float,
        soft_update_size : float,
        policy_regulariser : float,
        gradient_estimator : GradientEstimator,
        standardise_rewards : bool,
        pretrained_agents : Optional[ List[Agent] ] = None,
    ):
        self.n_agents = env.num_agents
        self.gamma = gamma
        # obs_dims = [obs.shape[0] for obs in env.observation_space]
        # act_dims = [act.n for act in env.action_space]
        obs_dims = [len(env.observation_space[0])] * self.n_agents
        act_dims = [len(env.action_space)] * self.n_agents
        self.agents = [
            Agent(
                agent_idx=ii,
                obs_dims=obs_dims,
                act_dims=act_dims,
                # TODO: Consider changing this to **config
                hidden_dim_width=hidden_dim_width,
                critic_lr=critic_lr,
                actor_lr=actor_lr,
                gradient_clip=gradient_clip,
                soft_update_size=soft_update_size,
                policy_regulariser=policy_regulariser,
                gradient_estimator=gradient_estimator,
            )
            for ii in range(self.n_agents)
        ] if pretrained_agents is None else pretrained_agents

        self.return_std = RunningMeanStd(shape=(self.n_agents,)) if standardise_rewards else None
        self.gradient_estimator = gradient_estimator  # Keep reference to GE object

    def acts(self, obs: List):
        actions = torch.tensor([self.agents[ii].act_behaviour(obs[ii]) for ii in range(self.n_agents)])
        return actions

    def update(self, sample):
        # sample['obs'] : agent batch obs
        batched_obs = torch.concat(sample['obs'], axis=1)
        batched_nobs = torch.concat(sample['nobs'], axis=1)

        # ********
        # TODO: This is all a bit cumbersome--could be cleaner?

        target_actions = [
            self.agents[ii].act_target(sample['nobs'][ii])
            for ii in range(self.n_agents)
        ]

        target_actions_one_hot = [
            F.one_hot(target_actions[ii], num_classes=self.agents[ii].n_acts)
            for ii in range(self.n_agents)
        ] # agent batch actions

        sampled_actions_one_hot = [
            F.one_hot(sample['acts'][ii].to(torch.int64), num_classes=self.agents[ii].n_acts)
            for ii in range(self.n_agents)
        ] # agent batch actions

        # ********
        # Standardise rewards if requested
        rewards = sample['rwds']
        if self.return_std is not None:
            self.return_std.update(rewards)
            rewards = ((rewards.T - self.return_std.mean) / torch.sqrt(self.return_std.var)).T
        # ********

        for ii, agent in enumerate(self.agents):
            agent.update_critic(
                all_obs=batched_obs,
                all_nobs=batched_nobs,
                target_actions_per_agent=target_actions_one_hot,
                sampled_actions_per_agent=sampled_actions_one_hot,
                rewards=rewards[ii].unsqueeze(dim=1),
                dones=sample['dones'][ii].unsqueeze(dim=1),
                gamma=self.gamma,
            )

            agent.update_actor(
                all_obs=batched_obs,
                agent_obs=sample['obs'][ii],
                sampled_actions=sampled_actions_one_hot,
            )

        for agent in self.agents:
            agent.soft_update()

        self.gradient_estimator.update_state() # Update GE state, if necessary

        return None

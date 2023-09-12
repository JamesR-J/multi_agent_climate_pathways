"""
@Theodore Wolf
A few simple networks that can be used by different types of agents
"""

import torch.nn as nn
import torch.nn.functional as F


class PolicyNet(nn.Module):
    """Outputs action preferences, to be used by Actor-critics"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(PolicyNet, self).__init__()
        self.layer = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(),
                                   nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                   )

        self.a = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs):
        l = self.layer(obs)
        out = self.a(l)

        return out


class Net(nn.Module):
    """Outputs Q-values for each action"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Net, self).__init__()

        self.layer = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(),
                                   nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                   )

        self.q = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs):
        l = self.layer(obs)
        q_values = self.q(l)

        return q_values


class DuellingNet(nn.Module):
    """Outputs duelling Q-values for each action"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(DuellingNet, self).__init__()

        self.layer = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(),
                                   nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                   #nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
                                   )

        self.a = nn.Linear(hidden_dim, action_dim)
        self.v = nn.Linear(hidden_dim, 1)

    def forward(self, obs):
        l = self.layer(obs)
        advantages = self.a(l)
        value = self.v(l)
        a_values = value + (advantages - advantages.mean())

        return a_values


class DualACNET(nn.Module):
    """Outputs both the value for the state and the action preferences"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(DualACNET, self).__init__()

        self.layer = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(),
                                   nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                   )

        self.dist = nn.Linear(hidden_dim, action_dim)
        self.v = nn.Linear(hidden_dim, 1)

    def forward(self, obs):
        l = self.layer(obs)
        policy_dist= F.softmax(self.dist(l), dim=-1)
        value = self.v(l)

        return value, policy_dist


class ActorNetwork(nn.Module):
    def __init__(self, obs_dim, hidden_dim_width, n_actions):
        super().__init__()
        self.obs_dim = obs_dim
        self.layers = nn.Sequential(*[
            nn.Linear(obs_dim, hidden_dim_width), nn.ReLU(),
            nn.Linear(hidden_dim_width, hidden_dim_width), nn.ReLU(),
            nn.Linear(hidden_dim_width, n_actions),
        ])

    def forward(self, obs):
        return self.layers(obs.float())  # TODO had to add float again lol idk why

    def hard_update(self, source):
        for target_param, source_param in zip(self.parameters(), source.parameters()):
            target_param.data.copy_(source_param.data)

    def soft_update(self, source, t):
        for target_param, source_param in zip(self.parameters(), source.parameters()):
            target_param.data.copy_((1 - t) * target_param.data + t * source_param.data)

class CriticNetwork(nn.Module):
    def __init__(self, all_obs_dims, all_acts_dims, hidden_dim_width):
        super().__init__()
        input_size = sum(all_obs_dims) + sum(all_acts_dims)

        self.layers = nn.Sequential(*[
            nn.Linear(input_size, hidden_dim_width),
            nn.ReLU(),
            nn.Linear(hidden_dim_width, hidden_dim_width),
            nn.ReLU(),
            nn.Linear(hidden_dim_width, 1),
        ])

    def forward(self, obs_and_acts):
        return self.layers(obs_and_acts) # TODO hmmm

    def hard_update(self, source):
        for target_param, source_param in zip(self.parameters(), source.parameters()):
            target_param.data.copy_(source_param.data)

    def soft_update(self, source, t):
        for target_param, source_param in zip(self.parameters(), source.parameters()):
            target_param.data.copy_((1 - t) * target_param.data + t * source_param.data)

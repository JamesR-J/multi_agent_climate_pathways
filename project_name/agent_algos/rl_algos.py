import sys
import time

import torch.nn as nn
import torch.nn.functional as F
from .gradient_estimators import *
from typing import List, Optional
from .utils import RunningMeanStd
from torch.optim import Adam
from copy import deepcopy
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def numpy_to_cuda(numpy_array):
    return torch.from_numpy(numpy_array).float().to(DEVICE)


class NN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.out = nn.Linear(hidden_dim // 2, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x.float()))
        x = F.relu(self.fc2(x))
        q_val = self.out(x)
        return q_val


class DuellingNN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(DuellingNN, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.a = nn.Linear(hidden_dim // 2, action_dim)
        self.v = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x.float()))
        x = F.relu(self.fc2(x))
        advantages = self.a(x)
        value = self.v(x)
        a_values = value + (advantages - advantages.mean())

        return a_values


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
        return self.layers(obs.float())

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
        return self.layers(obs_and_acts)

    def hard_update(self, source):
        for target_param, source_param in zip(self.parameters(), source.parameters()):
            target_param.data.copy_(source_param.data)

    def soft_update(self, source, t):
        for target_param, source_param in zip(self.parameters(), source.parameters()):
            target_param.data.copy_((1 - t) * target_param.data + t * source_param.data)


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
        if rewards.shape == (256, 1):  # 256 is batch size and hardcoded also in the NN
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


class D3QN(DQN):
    def __init__(self, state_dim, action_dim, lr=0.004133, rho=0.5307, tau=0.01856, **kwargs):
        super(D3QN, self).__init__(state_dim, action_dim, lr=lr, rho=rho, tau=tau, **kwargs)

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
        return "D3QN"


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ActorCritic, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def act(self, state):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy


def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        use_tanh = True
        use_orthogonal_init = True
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.activate_func = [nn.ReLU(), nn.Tanh()][use_tanh]  # Trick10: use tanh

        if use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3, gain=0.01)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        a_prob = torch.softmax(self.fc3(s), dim=1)
        return a_prob


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        use_tanh = True
        use_orthogonal_init = True
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.activate_func = [nn.ReLU(), nn.Tanh()][use_tanh]  # Trick10: use tanh

        if use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        v_s = self.fc3(s)
        return v_s


class PPO:
    def __init__(self, state_dim, action_dim, max_train_steps):
        self.batch_size = 2048
        self.mini_batch_size = 64
        self.lr_a = 3e-4
        self.lr_c = 3e-4
        self.gamma = 0.99
        self.lamda = 0.95
        self.epsilon = 0.2
        self.K_epochs = 10
        self.entropy_coef = 0.01
        self.set_adam_eps = True
        self.use_grad_clip = True
        self.use_lr_decay = True
        self.use_adv_norm = True

        self.max_train_steps = max_train_steps

        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim, action_dim)
        if self.set_adam_eps:  # Trick 9: set Adam epsilon=1e-5
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a, eps=1e-5)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c, eps=1e-5)
        else:
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)

    def evaluate(self, s):  # When evaluating the policy, we select the action with the highest probability
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        a_prob = self.actor(s).detach().numpy().flatten()
        a = np.argmax(a_prob)
        return a

    def choose_action(self, s, rational=False):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        with torch.no_grad():
            dist = Categorical(probs=self.actor(s))
            a = dist.sample()
            a_logprob = dist.log_prob(a)
        return a.numpy()[0], a_logprob.numpy()[0]

    def update(self, replay_buffer, total_steps):
        s, a, a_logprob, r, s_, dw, done = replay_buffer.numpy_to_tensor()  # Get training data
        """
            Calculate the advantage using GAE
            'dw=True' means dead or win, there is no next state s'
            'done=True' represents the terminal of an episode(dead or win or reaching the max_episode_steps). When calculating the adv, if done=True, gae=0
        """
        adv = []
        gae = 0
        with torch.no_grad():  # adv and v_target have no gradient
            vs = self.critic(s)
            vs_ = self.critic(s_)
            deltas = r + self.gamma # * (1.0 - dw) * vs_ - vs  # TODO removed the done feature, idk if right
            for delta, d in zip(reversed(deltas.flatten().numpy()), reversed(done.flatten().numpy())):
                gae = delta + self.gamma * self.lamda * gae # * (1.0 - d)  # TODO removed the done feature, idk if right
                adv.insert(0, gae)
            adv = torch.tensor(adv, dtype=torch.float).view(-1, 1)
            v_target = adv + vs
            if self.use_adv_norm:  # Trick 1:advantage normalization
                adv = ((adv - adv.mean()) / (adv.std() + 1e-5))

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Random sampling and no repetition. 'False' indicates that training will continue even if the number of samples in the last time is less than mini_batch_size
            for index in BatchSampler(SubsetRandomSampler(range(self.batch_size)), self.mini_batch_size, False):
                dist_now = Categorical(probs=self.actor(s[index]))
                dist_entropy = dist_now.entropy().view(-1, 1)  # shape(mini_batch_size X 1)
                a_logprob_now = dist_now.log_prob(a[index].squeeze()).view(-1, 1)  # shape(mini_batch_size X 1)
                # a/b=exp(log(a)-log(b))
                ratios = torch.exp(a_logprob_now - a_logprob[index])  # shape(mini_batch_size X 1)

                surr1 = ratios * adv[index]  # Only calculate the gradient of 'a_logprob_now' in ratios
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index]
                actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy  # shape(mini_batch_size X 1)
                # Update actor
                self.optimizer_actor.zero_grad()
                actor_loss.mean().backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.optimizer_actor.step()

                v_s = self.critic(s[index])
                critic_loss = F.mse_loss(v_target[index], v_s)
                # Update critic
                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer_critic.step()

        if self.use_lr_decay:  # Trick 6:learning rate Decay
            self.lr_decay(total_steps)

        return actor_loss.mean()

    def lr_decay(self, total_steps):
        lr_a_now = self.lr_a * (1 - total_steps / self.max_train_steps)
        lr_c_now = self.lr_c * (1 - total_steps / self.max_train_steps)
        for p in self.optimizer_actor.param_groups:
            p['lr'] = lr_a_now
        for p in self.optimizer_critic.param_groups:
            p['lr'] = lr_c_now


class Agent:
    def __init__(self,
                 agent_idx,
                 obs_dims,
                 act_dims,
                 hidden_dim_width,
                 critic_lr,
                 actor_lr,
                 gradient_clip,
                 soft_update_size,
                 policy_regulariser,
                 gradient_estimator,
                 ):
        self.agent_idx = agent_idx
        self.soft_update_size = soft_update_size
        self.n_obs = obs_dims[self.agent_idx]
        self.n_acts = act_dims[self.agent_idx]
        self.n_agents = len(obs_dims)
        self.gradient_clip = gradient_clip
        self.policy_regulariser = policy_regulariser
        self.gradient_estimator = gradient_estimator
        # -----------

        # ***** POLICY *****
        self.policy = ActorNetwork(self.n_obs, hidden_dim_width, self.n_acts)
        self.target_policy = ActorNetwork(self.n_obs, hidden_dim_width, self.n_acts)
        self.target_policy.hard_update(self.policy)
        # ***** ****** *****

        # ***** CRITIC *****
        self.critic = CriticNetwork(obs_dims, act_dims, hidden_dim_width)
        self.target_critic = CriticNetwork(obs_dims, act_dims, hidden_dim_width)
        self.target_critic.hard_update(self.critic)
        # ***** ****** *****

        # OPTIMISERS
        self.optim_actor = Adam(self.policy.parameters(), lr=actor_lr, eps=0.001)
        self.optim_critic = Adam(self.critic.parameters(), lr=critic_lr, eps=0.001)

    def act_behaviour(self, obs):
        policy_output = self.policy(Tensor(obs))
        gs_output = self.gradient_estimator(policy_output, need_gradients=False)
        return torch.argmax(gs_output, dim=-1)

    def act_target(self, obs):
        policy_output = self.target_policy(Tensor(obs))
        gs_output = self.gradient_estimator(policy_output, need_gradients=False)
        return torch.argmax(gs_output, dim=-1)

    def update_critic(self, all_obs, all_nobs, target_actions_per_agent, sampled_actions_per_agent, rewards, dones,
                      gamma):
        target_actions = torch.concat(target_actions_per_agent, axis=1)
        sampled_actions = torch.concat(sampled_actions_per_agent, axis=1)

        Q_next_target = self.critic(torch.concat((all_nobs, target_actions), dim=1))
        target_ys = rewards + (1 - dones) * gamma * Q_next_target
        behaviour_ys = self.critic(torch.concat((all_obs, sampled_actions), dim=1))

        loss = F.mse_loss(behaviour_ys, target_ys.detach())

        self.optim_critic.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.gradient_clip)
        self.optim_critic.step()

    def update_actor(self, all_obs, agent_obs, sampled_actions):
        policy_outputs = self.policy(agent_obs)
        gs_outputs = self.gradient_estimator(policy_outputs)

        _sampled_actions = deepcopy(sampled_actions)
        _sampled_actions[self.agent_idx] = gs_outputs

        loss = - self.critic(torch.concat((all_obs, *_sampled_actions), axis=1)).mean()
        loss += (policy_outputs ** 2).mean() * self.policy_regulariser

        self.optim_actor.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.gradient_clip)
        self.optim_actor.step()

    def soft_update(self):
        self.target_critic.soft_update(self.critic, self.soft_update_size)
        self.target_policy.soft_update(self.policy, self.soft_update_size)


class MADDPG:
    def __init__(
            self,
            env,
            critic_lr: float,
            actor_lr: float,
            gradient_clip: float,
            hidden_dim_width: int,
            gamma: float,
            soft_update_size: float,
            policy_regulariser: float,
            gradient_estimator: GradientEstimator,
            standardise_rewards: bool,
            pretrained_agents: Optional[List[Agent]] = None,
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

        target_actions = [
            self.agents[ii].act_target(sample['nobs'][ii])
            for ii in range(self.n_agents)
        ]

        target_actions_one_hot = [
            F.one_hot(target_actions[ii], num_classes=self.agents[ii].n_acts)
            for ii in range(self.n_agents)
        ]  # agent batch actions

        sampled_actions_one_hot = [
            F.one_hot(sample['acts'][ii].to(torch.int64), num_classes=self.agents[ii].n_acts)
            for ii in range(self.n_agents)
        ]  # agent batch actions

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

        self.gradient_estimator.update_state()  # Update GE state, if necessary

        return None

import time

import torch
import torch.nn as nn
import random
import wandb
from envs.AYS.AYS_Environment_MultiAgent import *
from envs.AYS.AYS_3D_figures import create_figure
# from envs.AYS.AYS_Environment import *
from learn_class import Learn
from learn import utils
import wandb
from rl_algos import DQN
from learn import agents as ag

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)


class MARL_agent:
    def __init__(self, num_agents=1, wandb_save=False, verbose=False, reward_type="PB",
                 max_episodes=5000, max_steps=500, max_frames=1e5,
                 max_epochs=50, seed=42, gamma=0.99, decay_number=0,
                 save_locally=False, animation=False):

        self.num_agents = num_agents

        self.env = AYS_Environment(num_agents=self.num_agents)
        self.state_dim = len(self.env.observation_space[0])
        self.action_dim = len(self.env.action_space)
        self.gamma = gamma

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.max_episodes = max_episodes
        self.max_steps = max_steps
        self.max_frames = max_frames
        self.max_epochs = max_epochs

        self.agent_str = "DQN"
        self.per_is = False
        self.alpha = 0.213
        self.buffer_size = 32768
        self.batch_size = 256
        self.decay_number = decay_number
        self.step_decay = int(self.max_frames / (self.decay_number + 1))

        # saving in wandb or logging
        self.wandb_save = wandb_save
        self.verbose = verbose
        self.save_locally = save_locally
        self.group_name = reward_type

        self.early_finish = [False] * self.num_agents

        # run information in a dictionary
        self.data = {'rewards': [],
                     'moving_avg_rewards': [],
                     'moving_std_rewards': [],
                     'frame_idx': 0,
                     'episodes': 0,
                     'final_point': []
                     }

        self.animation = animation

        self.agent_str = "DQN"

        self.agent = [DQN(self.state_dim, self.action_dim) for _ in range(self.num_agents)]

    def append_data(self, episode_reward):
        """We append the latest episode reward and calculate moving averages and moving standard deviations"""

        self.data['rewards'].append(episode_reward)

        concatenated = torch.stack(self.data['rewards'][-50:])  # TODO check this and the std and final states
        mean = torch.mean(concatenated, dim=0)
        self.data['moving_avg_rewards'].append(mean.tolist())

        std = torch.std(concatenated, dim=0)
        self.data['moving_avg_rewards'].append(std.tolist())

        self.data['episodes'] += 1

        final_states = [self.env.which_final_state(agent).name for agent in range(self.num_agents)]
        self.data['final_point'].append(final_states)

        # we log or print depending on settings
        wandb.log({'episode_reward': episode_reward,
                   "moving_average": self.data['moving_avg_rewards'][-1]}) \
            if self.wandb_save else None

        print("Episode:", self.data['episodes'], "|| Reward:", round(episode_reward), "|| Final State ",
              self.env.which_final_state().name) \
            if self.verbose else None

    def training_run(self):

        if self.animation:
            plt.ion()
            fig, ax3d = create_figure()
            colors = plt.cm.brg(np.linspace(0, 1, self.num_agents))
            plt.show()

        self.data['frame_idx'] = self.data['episodes'] = 0

        config = None
        if self.wandb_save:
            # wandb.init(project="ucl_diss", entity="climate_policy_optim", config=config, job_type=str(experiment.agent),group=group_name)
            wandb.init(project="ucl_diss", config=None)

        # wandb.init(
        #     project="ucl_diss",
        #     config={
        #         "learning_rate": 0.02,
        #         "architecture": "CNN",
        #         "dataset": "CIFAR-100",
        #         "epochs": 10,
        #     }
        # )

        # memory = utils.PER_IS_ReplayBuffer(self.buffer_size, alpha=self.alpha,
        #                                    state_dim=self.state_dim) if self.per_is else utils.ReplayBuffer(self.buffer_size)
        memory = [utils.ReplayBuffer(self.buffer_size) for _ in range(self.num_agents)]

        for episodes in range(self.max_episodes):

            # reset environment to a random state (0.5, 0.5, 0.5) * gaussian noise
            state_n = self.env.reset()
            self.early_finish = [False] * self.num_agents
            episode_reward = torch.tensor([0.0]).repeat(self.num_agents, 1)

            if self.animation:
                a_values = [[] for _ in range(self.num_agents)]
                y_values = [[] for _ in range(self.num_agents)]
                s_values = [[] for _ in range(self.num_agents)]

            for i in range(self.max_steps):

                action_n = torch.tensor([self.agent[ind].get_action(state_n[ind]) for ind in range(self.num_agents)])
                # action_n = torch.tensor([0 for _ in range(self.num_agents)])

                # step through environment
                next_state, reward, done, _ = self.env.step(action_n)

                for agent in range(self.num_agents):
                    if not self.early_finish[agent]:
                        episode_reward[agent] += reward[agent]
                        memory[agent].push(state_n[agent], action_n[agent], reward[agent], next_state[agent], done[agent])
                        if len(memory[agent]) > self.batch_size:
                            sample = memory[agent].sample(self.batch_size)  # (5,128,xxx)
                            loss, _ = self.agent[agent].update(sample)
                            label = "loss_agent_" + str(agent)
                            wandb.log({label: loss}) if self.wandb_save else None
                        if done[agent]:
                            self.early_finish[agent] = True
                    # else:  # TODO allows another agent to carry on running even if one has reached final state
                    #     next_state[agent] = state_n[agent].clone()

                state_n = next_state.clone()
                self.data['frame_idx'] += 1

                if self.animation:
                    for j in range(self.num_agents):
                        a_values[j].append(state_n[j][0])
                        y_values[j].append(state_n[j][1])
                        s_values[j].append(state_n[j][2])

                    for agent in range(self.num_agents):
                        if self.early_finish[agent]:
                            ax3d.plot3D(xs=a_values[agent], ys=y_values[agent], zs=s_values[agent],
                                        color='r', alpha=0.3, lw=3)
                        else:
                            ax3d.plot3D(xs=a_values[agent], ys=y_values[agent], zs=s_values[agent],
                                    color=colors[agent], alpha=0.3, lw=3)

                    # ax3d.legend([f'Agent {j + 1}' for j in range(experiment.env.num_agents)])

                    if i == 0:
                        ays_plot.plot_hairy_lines(100, ax3d)
                    plt.pause(0.001)

                if torch.all(done):
                    break

            if self.animation:
                ax3d.clear()
                fig, ax3d = create_figure(reset=True, fig3d=fig, ax3d=ax3d)

            self.append_data(episode_reward)

        for agent in range(self.num_agents):
            success_rate = np.stack(self.data["final_point"])[:, agent].tolist().count("GREEN_FP") / self.data["episodes"]
            print("Agent_" + str(agent) + " Success rate: ", round(success_rate, 3))

            if self.wandb_save:
                wandb.run.summary["Agent_" + str(agent) + "_mean_reward"] = np.mean(np.stack(self.data['rewards'])[:, agent])
                wandb.run.summary["Agent_" + str(agent) + "_top_reward"] = max(np.stack(self.data['rewards'])[:, agent])
                wandb.run.summary["Agent_" + str(agent) + "_success_rate"] = success_rate

        if self.wandb_save:
            wandb.run.summary["data"] = self.data
            wandb.finish()

        if self.animation:
            plt.ioff()

# if __name__ == "__main__":
#     return
import random
import time
from datetime import datetime

from envs.AYS_Environment_MultiAgent import *
from envs.graph_functions import create_figure_ays, create_figure_ricen
from learn import utils
import wandb
from rl_algos import DQN, DuelDDQN
from envs.ricen import RiceN

import matplotlib.pyplot as plt

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)


class MARL_agent:
    def __init__(self, model, chkpt_load_path, num_agents=1, wandb_save=False, verbose=False, reward_type="PB",
                 max_episodes=4000, max_steps=1000, max_frames=1e5,
                 max_epochs=50, seed=42, gamma=0.99, decay_number=0,
                 save_locally=False, animation=False, test_actions=False, top_down=False, chkpt_load=False,
                 obs_type='all_shared', load_multi=False, rational=[True, True]):

        self.num_agents = num_agents
        self.obs_type = obs_type

        self.model = model

        self.reward_type = [reward_type] * self.num_agents
        self.reward_type = reward_type

        assert self.num_agents == len(self.reward_type), "Reward function number does not match no. of agents"

        self.gamma = gamma

        if self.model == "ays":
            self.env = AYS_Environment(num_agents=self.num_agents, reward_type=self.reward_type, max_steps=max_steps,
                                       gamma=self.gamma, obs_type=self.obs_type)
        elif self.model == "rice-n":
            self.env = RiceN(num_agents=self.num_agents, episode_length=max_steps, reward_type=self.reward_type,
                             max_steps=max_steps, discount=self.gamma)
        self.state_dim = len(self.env.observation_space[0])
        self.action_dim = len(self.env.action_space)

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.max_episodes = max_episodes
        self.max_steps = max_steps
        self.max_frames = max_frames
        self.max_epochs = max_epochs

        self.alpha = 0.213
        self.beta = 0.7389
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
        self.top_down = top_down

        # self.agent_str = "DQN"
        self.agent_str = "DuelDDQN"

        self.chkpt_load = chkpt_load
        self.chkpt_load_path = chkpt_load_path
        self.load_multi = load_multi
        self.episode_count = 0

        if self.chkpt_load:
            self.episode_count = torch.load(self.chkpt_load_path)['episode_count']
            self.max_episodes += self.episode_count

        if self.episode_count > 0:
            epsilon = 0.99
        else:
            epsilon = 1.0

        self.rational = rational
        self.rational_ep = [True] * self.num_agents
        self.irrational_end = [0] * self.num_agents
        self.rational_const = 0.2  # TODO tweak this param aswell for rationality
        print("Rational behaviour: {}".format(self.rational))

        if self.agent_str == "DQN":
            self.agent = [DQN(self.state_dim, self.action_dim, epsilon=epsilon) for _ in range(self.num_agents)]
        elif self.agent_str == "DuelDDQN":
            self.agent = [DuelDDQN(self.state_dim, self.action_dim, epsilon=epsilon) for _ in range(self.num_agents)]

        self.test_actions = test_actions

    def append_data(self, episode_reward):
        """We append the latest episode reward and calculate moving averages and moving standard deviations"""

        self.data['rewards'].append(episode_reward)

        concatenated = torch.stack(self.data['rewards'][-50:])

        mean = torch.mean(concatenated, dim=0)
        self.data['moving_avg_rewards'].append(mean.tolist())

        std = torch.std(concatenated, dim=0)
        self.data['moving_std_rewards'].append(std.tolist())

        self.data['episodes'] += 1

        if self.model == 'ays':  # TODO redo this for ricen
            final_states = [self.env.which_final_state(agent).name for agent in range(self.num_agents)]
            self.data['final_point'].append(final_states)

        # we log or print depending on settings
        if self.wandb_save:
            for agent in range(self.num_agents):
                label_1 = "episode_reward_agent_" + str(agent)
                label_2 = "moving_avg_rewards_agent_" + str(agent)
                wandb.log({label_1: episode_reward[agent]})
                wandb.log({label_2: self.data['moving_avg_rewards'][-1][agent][
                    0]})

        if self.model == 'ays':  # TODO redo this for ricen
            print("Episode:", self.data['episodes'], "|| Reward:", round(episode_reward), "|| Final State ",
                  self.env.which_final_state().name) \
                if self.verbose else None

    def rational_calc(self, agent, step):
        if not self.rational[agent]:
            if self.rational_ep[agent]:
                if np.random.uniform() < self.rational_const:
                    self.rational_ep[agent] = False
                    self.irrational_end[agent] = step + np.random.randint(1, 10)  # TODO adjust this range
            else:
                if step == self.irrational_end[agent]:
                    self.rational_ep[agent] = True

    def training_run(self):

        if self.animation:
            plt.ion()
            if self.model == 'ays':
                fig, ax3d = create_figure_ays(top_down=self.top_down)
            if self.model == 'rice-n':
                fig, ax3d = create_figure_ricen(top_down=self.top_down)
            colors = plt.cm.brg(np.linspace(0, 1, self.num_agents))
            plt.show()

        self.data['frame_idx'] = self.data['episodes'] = 0

        if self.wandb_save:
            wandb.init(project="ucl_diss", group=self.model,
                       config={"reward_type": self.reward_type,
                               "observations": self.obs_type,
                               "rational": self.rational
                               })

        if self.agent_str == "DuelDDQN":
            memory = [utils.PER_IS_ReplayBuffer(self.buffer_size, alpha=self.alpha, state_dim=self.state_dim) for _ in range(self.num_agents)]
        else:
            memory = [utils.ReplayBuffer(self.buffer_size) for _ in range(self.num_agents)]

        if self.chkpt_load:
            checkpoint = torch.load(self.chkpt_load_path)
            for agent in range(self.num_agents):
                if self.load_multi:
                    chkpt_string = 'agent_0'
                else:
                    chkpt_string = 'agent_' + str(agent)  # TODO maybe add a check here so cant load a single agent into a multi agent unless self.load_multi is true, and then vice versa or so
                self.agent[agent].target_net.load_state_dict(checkpoint[chkpt_string + '_target_state_dict'])
                self.agent[agent].policy_net.load_state_dict(checkpoint[chkpt_string + '_policy_state_dict'])
                self.agent[agent].optimizer.load_state_dict(checkpoint[chkpt_string + '_optimiser_state_dict'])
            self.episode_count = checkpoint['episode_count']

        for episodes in range(self.max_episodes):

            # reset environment to a random state (0.5, 0.5, 0.5) * gaussian noise
            state_n, obs_n = self.env.reset()  # TODO check this stuff
            self.early_finish = [False] * self.num_agents
            episode_reward = torch.tensor([0.0]).repeat(self.num_agents, 1)

            self.rational_ep = [True] * self.num_agents
            self.irrational_end = [0] * self.num_agents

            if self.animation:
                x_values = [[] for _ in range(self.num_agents)]
                y_values = [[] for _ in range(self.num_agents)]
                z_values = [[] for _ in range(self.num_agents)]

            for i in range(self.max_steps):

                for agent in range(self.num_agents):
                    self.rational_calc(agent, i)

                action_n = torch.tensor([self.agent[agent].get_action(obs_n[agent], rational=self.rational_ep[agent]) for agent in range(self.num_agents)])
                if self.test_actions:
                    action_n = torch.tensor([0 for _ in range(self.num_agents)])

                # step through environment
                next_state, reward, done, next_obs = self.env.step(action_n)

                for agent in range(self.num_agents):
                    # if not self.early_finish[agent]:
                        episode_reward[agent] += reward[agent]
                        memory[agent].push(obs_n[agent], action_n[agent], reward[agent], next_obs[agent], done[agent])
                        if len(memory[agent]) > self.batch_size:
                            if self.agent_str == "DuelDDQN":
                                self.beta = 1 - (1 - self.beta) * np.exp(-0.05 * episodes)
                                sample = memory[agent].sample(self.batch_size, self.beta)
                                loss, tds = self.agent[agent].update(
                                    (sample['obs'], sample['action'], sample['reward'], sample['next_obs'],
                                     sample['done']),
                                    weights=sample['weights']
                                )
                                new_tds = np.abs(tds.cpu().numpy()) + 1e-6
                                memory[agent].update_priorities(sample['indexes'], new_tds)
                            else:
                                sample = memory[agent].sample(self.batch_size)  # shape(5,128,xxx)
                                loss, _ = self.agent[agent].update(sample)

                            label = "loss_agent_" + str(agent)
                            wandb.log({label: loss}) if self.wandb_save else None
                        if done[agent]:
                            self.early_finish[agent] = True
                    # else:  # TODO allows another agent to carry on running even if one has reached final state
                    #     next_state[agent] = state_n[agent].clone()

                obs_n = next_obs.clone()
                state_n = next_state.clone()
                self.data['frame_idx'] += 1

                if self.animation:
                    for line in ax3d.lines:
                        line.remove()

                    for agent in range(self.num_agents):
                        x_values[agent].append(state_n[agent][0])
                        y_values[agent].append(state_n[agent][1])
                        z_values[agent].append(state_n[agent][2])

                        if self.early_finish[agent]:
                            ax3d.plot3D(xs=x_values[agent], ys=y_values[agent], zs=z_values[agent],
                                        color='r', alpha=0.8, lw=3, label="Agent : {}".format(agent))
                        else:
                            ax3d.plot3D(xs=x_values[agent], ys=y_values[agent], zs=z_values[agent],
                                        color=colors[agent], alpha=0.8, lw=3, label="Agent : {}".format(agent))

                    ax3d.set_title(["Agent {} {} reward : {:.2f}".format(agent, self.reward_type[agent], reward[agent][0]) for agent in
                                    range(self.num_agents)])
                    # ax3d.set_title(i)
                    ax3d.legend()

                    plt.pause(0.00001)

                if torch.all(done):
                # if torch.any(done):  # TODO but means that the early ended agent won't get more reward as doesnt get final count function thing - see if it matters I guess
                    break

            if self.animation:
                ax3d.clear()
                if self.model == 'ays':
                    fig, ax3d = create_figure_ays(reset=True, fig3d=fig, ax3d=ax3d, top_down=self.top_down)
                if self.model == 'rice-n':
                    fig, ax3d = create_figure_ricen(reset=True, fig3d=fig, ax3d=ax3d, top_down=self.top_down)

            self.append_data(episode_reward)

        self.episode_count += self.max_episodes

        # add checkpointing here
        chkpt_save_path = './checkpoints/env=' + str(self.model) + '_reward_type=' + str(self.reward_type) \
                          + '_obs_type=' + str(self.obs_type) + '_num_agents=' + str(self.num_agents) \
                          + '_episodes=' + str(self.episode_count)
        tot_dict = {}
        for agent in range(self.num_agents):
            tot_dict['agent_' + str(agent) + '_target_state_dict'] = self.agent[agent].target_net.state_dict()
            tot_dict['agent_' + str(agent) + '_policy_state_dict'] = self.agent[agent].policy_net.state_dict()
            tot_dict['agent_' + str(agent) + '_optimiser_state_dict'] = self.agent[agent].optimizer.state_dict()
        tot_dict['episode_count'] = self.episode_count
        if not os.path.exists(chkpt_save_path):
            os.makedirs(chkpt_save_path)
        torch.save(tot_dict,
                   chkpt_save_path + '/end_time=' + str(datetime.now().strftime("%d-%m-%Y_%H-%M-%S")) + '.tar')

        for agent in range(self.num_agents):
            success_rate = np.stack(self.data["final_point"])[:, agent].tolist().count("GREEN_FP") / self.data[
                "episodes"]
            print("Agent_" + str(agent) + " Success rate: ", round(success_rate, 3))

            if self.wandb_save:
                wandb.run.summary["Agent_" + str(agent) + "_mean_reward"] = np.mean(
                    np.stack(self.data['rewards'])[:, agent])
                wandb.run.summary["Agent_" + str(agent) + "_top_reward"] = max(np.stack(self.data['rewards'])[:, agent])
                wandb.run.summary["Agent_" + str(agent) + "_success_rate"] = success_rate

        if self.wandb_save:
            wandb.run.summary["data"] = self.data
            wandb.finish()

        if self.animation:
            plt.ioff()

# if __name__ == "__main__":
#     return

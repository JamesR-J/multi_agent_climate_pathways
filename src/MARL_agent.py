from envs.AYS.AYS_Environment import *
import torch
import random
import wandb
from envs.AYS.AYS_Environment_MultiAgent import *
from envs.AYS.AYS_3D_figures import create_figure
# from envs.AYS.AYS_Environment import *
from learn_class import Learn
from learn import utils
import wandb
from learn import agents as ag

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


print(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))


class MARL_agent:
    def __init__(self, num_agents=1, wandb_save=False, verbose=False, reward_type="PB",
                 max_episodes=2000, max_steps=600, max_frames=1e5,
                 max_epochs=50, seed=42, gamma=0.99, decay_number=0,
                 save_locally=False, animation=False):

        self.num_agents = num_agents

        self.env = AYS_Environment(num_agents=self.num_agents)
        self.state_dim = len(self.env.observation_space)
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
        # self.agent = "DQN"

        try:
            self.agent = eval("ag." + self.agent_str)(self.state_dim, self.action_dim,
                                                 gamma=self.gamma, step_decay=self.step_decay)
        except:
            print('Not a valid agent, try "Random", "A2C", "DQN", "PPO" or "DuelDDQN".')

    def training_run(self):
        state = [0.5, 0.5, 0.5]

        if self.animation:
            plt.ion()
            fig, ax3d = create_figure()
            colors = plt.cm.brg(np.linspace(0, 1, self.num_agents))
            plt.show()

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

        memory = utils.PER_IS_ReplayBuffer(self.buffer_size, alpha=self.alpha,
                                           state_dim=self.state_dim) if self.per_is else utils.ReplayBuffer(self.buffer_size)

        for episodes in range(100):

            # reset environment to a random state (0.5, 0.5, 0.5) * gaussian noise
            state = self.env.reset()
            episode_reward = 0

            if self.animation:
                a_values = [[] for _ in range(self.num_agents)]
                y_values = [[] for _ in range(self.num_agents)]
                s_values = [[] for _ in range(self.num_agents)]

            for i in range(1000):

                # get state
                if self.num_agents == 1:
                    state = state.tolist()[0]

                action = self.agent.get_action(state)  # TODO reinstate this once got env to work can figure out the MARL algo
                # action = torch.randint(0, 3, (experiment.env.num_agents, 1))

                # step through environment
                next_state, reward, done, _ = self.env.step(action)

                if self.num_agents == 1:
                    reward = reward.tolist()[0][0]
                    next_state = next_state.tolist()[0]  # TODO make it work better with MARL then this shit to list stuff

                episode_reward += reward

                memory.push(state, action, reward, next_state, done)
                if len(memory) > self.batch_size:
                    # if we are using prioritised experience replay buffer with importance sampling
                    if self.per_is:
                        beta = 1 - (1 - beta) * np.exp(
                            -0.05 * episodes)  # we converge beta to 1, using episodes is flawed, use frames instead
                        sample = memory.sample(self.batch_size, beta)
                        loss, tds = self.agent.update(
                            (sample['obs'], sample['action'], sample['reward'], sample['next_obs'], sample['done']),
                            weights=sample['weights']
                        )
                        new_tds = np.abs(tds.cpu().numpy()) + 1e-6
                        memory.update_priorities(sample['indexes'], new_tds)
                    # otherwise we just uniformly sample
                    else:
                        sample = memory.sample(self.batch_size)  # (5,128,xxx)
                        # print(sample)
                        # print(np.array(sample).shape)
                        loss, _ = self.agent.update(sample)
                    wandb.log({'loss': loss}) if self.wandb_save else None

                state = next_state

                if self.num_agents == 1:
                    state = torch.tensor([state])

                if self.animation:
                    for j in range(self.num_agents):
                        a_values[j].append(state[j][0])
                        y_values[j].append(state[j][1])
                        s_values[j].append(state[j][2])

                    for j in range(self.num_agents):
                        ax3d.plot3D(xs=a_values[j], ys=y_values[j], zs=s_values[j],
                                    color=colors[j], alpha=0.3, lw=3)

                    # ax3d.legend([f'Agent {j + 1}' for j in range(experiment.env.num_agents)])

                    if i == 0:
                        ays_plot.plot_hairy_lines(100, ax3d)
                    plt.pause(0.001)

                    # if the episode is finished we stop there
                    if done:
                        break
            if self.animation:
                ax3d.clear()
                fig, ax3d = create_figure(reset=True, fig3d=fig, ax3d=ax3d)

        if self.wandb_save:
            # wandb.run.summary["mean_reward"] = np.mean(self.data['rewards'])
            # wandb.run.summary["top_reward"] = max(self.data['rewards'])
            # wandb.run.summary["success_rate"] = success_rate
            # wandb.run.summary["data"] = self.data
            wandb.finish()

        if self.animation:
            plt.ioff()

# if __name__ == "__main__":
#     return
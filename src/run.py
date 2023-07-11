from envs.AYS.AYS_Environment_MultiAgent import *
from envs.AYS.AYS_3D_figures import create_figure
# from envs.AYS.AYS_Environment import *
from learn_class import Learn
from learn import utils
import wandb

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from MARL_agent import MARL_agent



# class PB_Learn(Learn):
#     def __init__(self, **kwargs):
#         super(PB_Learn, self).__init__(**kwargs)
#         self.num_agents = 1
#         self.env = AYS_Environment(num_agents=self.num_agents, **kwargs)


if __name__ == "__main__":
    # TODO add argparse here and stuff ya know
    # animation = False
    animation = True
    # wandb_save = False
    wandb_save = True
    num_agents = 2
    marl_agent = MARL_agent(num_agents=num_agents, animation=animation, wandb_save=wandb_save)

    marl_agent.training_run()
    # marl_agent.env.test_reward_functions()


    # animation = False
    # # animation = True
    # experiment = PB_Learn(wandb_save=False, reward_type="PB", verbose=True)
    # experiment.set_agent("DQN")
    #
    # state = [0.5, 0.5, 0.5]
    #
    # if animation:
    #     plt.ion()
    #     fig, ax3d = create_figure()
    #     colors = plt.cm.brg(np.linspace(0, 1, experiment.env.num_agents))
    #     plt.show()
    #
    # agent_str = "DQN"
    # per_is = False
    # state_dim = len(experiment.env.observation_space)
    # action_dim = len(experiment.env.action_space)
    # alpha = 0.213
    # buffer_size = 32768
    # batch_size = 256
    # gamma = 0.99
    # decay_number = 0
    # max_frames = 1e5
    # step_decay = int(max_frames / (decay_number + 1))
    #
    # config = None
    # wandb_save = True
    # group_name = "PB"
    # if wandb_save:
    #     # wandb.init(project="ucl_diss", entity="climate_policy_optim", config=config, job_type=str(experiment.agent),group=group_name)
    #     wandb.init(project="ucl_diss", config=None)
    #
    #
    # # wandb.init(
    # #     project="ucl_diss",
    # #     config={
    # #         "learning_rate": 0.02,
    # #         "architecture": "CNN",
    # #         "dataset": "CIFAR-100",
    # #         "epochs": 10,
    # #     }
    # # )
    #
    # memory = utils.PER_IS_ReplayBuffer(buffer_size, alpha=alpha,
    #                                         state_dim=state_dim) if per_is else utils.ReplayBuffer(buffer_size)
    #
    # for episodes in range(100):
    #
    #     # reset environment to a random state (0.5, 0.5, 0.5) * gaussian noise
    #     state = experiment.env.reset()
    #     episode_reward = 0
    #
    #     if animation:
    #         a_values = [[] for _ in range(experiment.env.num_agents)]
    #         y_values = [[] for _ in range(experiment.env.num_agents)]
    #         s_values = [[] for _ in range(experiment.env.num_agents)]
    #
    #     for i in range(1000):
    #
    #         # get state
    #         if experiment.env.num_agents == 1:
    #             state = state.tolist()[0]
    #
    #         action = experiment.agent.get_action(state)  # TODO reinstate this once got env to work can figure out the MARL algo
    #         # action = torch.randint(0, 3, (experiment.env.num_agents, 1))
    #
    #         # step through environment
    #         next_state, reward, done, _ = experiment.env.step(action)
    #
    #         if experiment.env.num_agents == 1:
    #             reward = reward.tolist()[0][0]
    #             next_state = next_state.tolist()[0]  # TODO make it work better with MARL then this shit to list stuff
    #
    #         episode_reward += reward
    #
    #         memory.push(state, action, reward, next_state, done)
    #         if len(memory) > batch_size:
    #             # if we are using prioritised experience replay buffer with importance sampling
    #             if per_is:
    #                 beta = 1 - (1 - beta) * np.exp(
    #                     -0.05 * episodes)  # we converge beta to 1, using episodes is flawed, use frames instead
    #                 sample = memory.sample(batch_size, beta)
    #                 loss, tds = experiment.agent.update(
    #                     (sample['obs'], sample['action'], sample['reward'], sample['next_obs'], sample['done']),
    #                     weights=sample['weights']
    #                 )
    #                 new_tds = np.abs(tds.cpu().numpy()) + 1e-6
    #                 memory.update_priorities(sample['indexes'], new_tds)
    #             # otherwise we just uniformly sample
    #             else:
    #                 sample = memory.sample(batch_size)  # (5,128,xxx)
    #                 # print(sample)
    #                 # print(np.array(sample).shape)
    #                 loss, _ = experiment.agent.update(sample)
    #             wandb.log({'loss': loss}) if wandb_save else None
    #
    #         state = next_state
    #
    #         # print(state)
    #
    #         if experiment.env.num_agents == 1:
    #             state = torch.tensor([state])
    #
    #         if animation:
    #             for j in range(experiment.env.num_agents):
    #                 a_values[j].append(state[j][0])
    #                 y_values[j].append(state[j][1])
    #                 s_values[j].append(state[j][2])
    #
    #             for j in range(experiment.env.num_agents):
    #                 ax3d.plot3D(xs=a_values[j], ys=y_values[j], zs=s_values[j],
    #                             color=colors[j], alpha=0.3, lw=3)
    #
    #             # ax3d.legend([f'Agent {j + 1}' for j in range(experiment.env.num_agents)])
    #
    #             if i == 0:
    #                 ays_plot.plot_hairy_lines(100, ax3d)
    #             plt.pause(0.001)
    #
    #             # if the episode is finished we stop there
    #             if done:
    #                 break
    #     if animation:
    #         ax3d.clear()
    #         fig, ax3d = create_figure(reset=True, fig3d=fig, ax3d=ax3d)
    #
    # if wandb_save:
    #     # wandb.run.summary["mean_reward"] = np.mean(self.data['rewards'])
    #     # wandb.run.summary["top_reward"] = max(self.data['rewards'])
    #     # wandb.run.summary["success_rate"] = success_rate
    #     # wandb.run.summary["data"] = self.data
    #     wandb.finish()
    #
    # if animation:
    #     plt.ioff()

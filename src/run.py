from envs.AYS.AYS_Environment_MultiAgent import *
from envs.AYS.AYS_3D_figures import create_figure
# from envs.AYS.AYS_Environment import *
from learn_class import Learn

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class PB_Learn(Learn):
    def __init__(self, **kwargs):
        super(PB_Learn, self).__init__(**kwargs)
        self.num_agents = 10
        self.env = AYS_Environment(num_agents=self.num_agents, **kwargs)

if __name__ == "__main__":
    # animation = False
    animation = True
    experiment = PB_Learn(wandb_save=False, reward_type="PB", verbose=True)
    experiment.set_agent("DuelDDQN")
    # experiment.learning_loop_offline(128, 2 ** 13, per_is=True)

    state = [0.5, 0.5, 0.5]

    if animation:
        plt.ion()
        fig, ax3d = create_figure()
        colors = plt.cm.brg(np.linspace(0, 1, experiment.env.num_agents))
        plt.show()

    for episodes in range(1):

        # reset environment to a random state (0.5, 0.5, 0.5) * gaussian noise
        state = experiment.env.reset()
        episode_reward = 0

        if animation:
            a_values = [[] for _ in range(experiment.env.num_agents)]
            y_values = [[] for _ in range(experiment.env.num_agents)]
            s_values = [[] for _ in range(experiment.env.num_agents)]

        for i in range(1000):

            # get state
            # action = experiment.agent.get_action(state)  # TODO reinstate this once got env to work can figure out the MARL algo
            action = torch.randint(0, 3, (experiment.env.num_agents, 1))

            # step through environment
            next_state, reward, done, _ = experiment.env.step(action)

            # add reward
            episode_reward += reward
            # prepare for next iteration
            state = next_state

            # print(state)

            if animation:
                for j in range(experiment.env.num_agents):
                    a_values[j].append(state[j][0])
                    y_values[j].append(state[j][1])
                    s_values[j].append(state[j][2])

                for j in range(experiment.env.num_agents):
                    ax3d.plot3D(xs=a_values[j], ys=y_values[j], zs=s_values[j],
                                color=colors[j], alpha=0.3, lw=3)

                # ax3d.legend([f'Agent {j + 1}' for j in range(experiment.env.num_agents)])

                if i == 0:
                    ays_plot.plot_hairy_lines(100, ax3d)
                plt.pause(0.001)

                # if the episode is finished we stop there
                if done:
                    break
        if animation:
            ax3d.clear()
            fig, ax3d = create_figure(reset=True, fig3d=fig, ax3d=ax3d)
    if animation:
        plt.ioff()

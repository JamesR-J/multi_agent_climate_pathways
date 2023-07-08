from envs.AYS.AYS_Environment_MultiAgent import *
# from envs.AYS.AYS_Environment import *
from learn_class import Learn

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class PB_Learn(Learn):
    def __init__(self, **kwargs):
        super(PB_Learn, self).__init__(**kwargs)
        # self.num_agents = 2
        self.env = AYS_Environment(**kwargs)

if __name__ == "__main__":
    num_agents = 2
    experiment = PB_Learn(wandb_save=False, reward_type="PB", verbose=True)
    experiment.set_agent("DuelDDQN")
    # experiment.learning_loop_offline(128, 2 ** 13, per_is=True)

    state = [0.5, 0.5, 0.5]

    for episodes in range(1):

        # reset environment to a random state (0.5, 0.5, 0.5) * gaussian noise
        state = experiment.env.reset()
        episode_reward = 0

        plt.ion()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        x_values = []
        y_values = []
        z_values = []

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('RL Agent State')

        plot = ax.plot([], [], [], color='b')[0]

        plt.show()

        for i in range(1000):

            # get state
            # action = experiment.agent.get_action(state)  # TODO reinstate this once got env to work can figure out the MARL algo
            action = (0, 0)
            # print(action)

            # step through environment
            next_state, reward, done, _ = experiment.env.step(action)

            # add reward
            episode_reward += reward
            # prepare for next iteration
            state = next_state

            # print(state)

            x_values.append(state[0, 0])
            y_values.append(state[0, 1])
            z_values.append(state[0, 2])

            # Update the plot
            plot.set_data(x_values, y_values)
            plot.set_3d_properties(z_values)

            # Adjust the plot limits (if needed)
            ax.relim()
            ax.autoscale_view()

            # Pause to allow the plot to update
            plt.pause(0.001)

            # if the episode is finished we stop there
            if done:
                break

        plt.ioff()
        plt.show()


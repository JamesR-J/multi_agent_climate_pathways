from envs.AYS.AYS_Environment_MultiAgent import *
from learn_class import Learn

class Markov_Learn(Learn):
    def __init__(self, **kwargs):
        super(Markov_Learn, self).__init__(**kwargs)
        self.env = velocity_AYS(**kwargs)
        self.state_dim = len(self.env.observation_space) * 2
        self.group_name = "Markov"

if __name__ == "__main__":
    experiment = Markov_Learn(wandb_save=False, reward_type="PB", verbose=True)
    experiment.set_agent("DuelDDQN")
    # experiment.learning_loop_offline(128, 2 ** 13, per_is=True)

    state = [0.5, 0.5, 0.5]

    for episodes in range(1):

        # reset environment to a random state (0.5, 0.5, 0.5) * gaussian noise
        state = experiment.env.reset()
        episode_reward = 0

        for i in range(10):

            # get state
            action = experiment.agent.get_action(state)
            # print(action)

            # step through environment
            next_state, reward, done, _ = experiment.env.step(action)

            # add reward
            episode_reward += reward
            # prepare for next iteration
            state = next_state

            # if the episode is finished we stop there
            if done:
                break


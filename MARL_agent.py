import random
from datetime import datetime
import time
from matplotlib.animation import FuncAnimation, PillowWriter

from envs.AYS_Environment_MultiAgent import *
from envs.graph_functions import create_figure_ays, create_figure_ricen
from agent_algos import utils
import wandb
from agent_algos.rl_algos import DQN, D3QN, MADDPG, PPO
# from envs.ricen import RiceN
import agent_algos.gradient_estimators

import matplotlib.pyplot as plt

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)


class MARL_agent:
    def __init__(self, model, chkpt_load_path, num_agents=1, wandb_save=False, verbose=False, reward_type="PB",
                 max_episodes=2000, max_steps=800, max_frames=1e5,
                 max_epochs=50, seed=42, gamma=0.99, decay_number=0,
                 save_locally=False, animation=False, test_actions=False, top_down=False, chkpt_load=False,
                 obs_type='all_shared', load_multi=False, rational=[True, True], trade_actions=False, algorithm="PPO",
                 homogeneous=False, rational_choice="2nd_best"):
        self.homogeneous = homogeneous

        print(f"Homogeneous : {self.homogeneous}")

        self.num_agents = num_agents
        self.obs_type = obs_type

        self.model = model

        self.reward_type = [reward_type] * self.num_agents
        self.reward_type = reward_type

        assert self.num_agents == len(self.reward_type), "Reward function number does not match no. of agents"

        self.gamma = gamma

        if self.model == "ays":
            self.env = AYS_Environment(num_agents=self.num_agents, reward_type=self.reward_type, max_steps=max_steps,
                                       gamma=self.gamma, obs_type=self.obs_type, trade_actions=trade_actions,
                                       homogeneous=self.homogeneous)
        elif self.model == "rice-n":
            sys.exit(0)
            # self.env = RiceN(num_agents=self.num_agents, episode_length=max_steps, reward_type=self.reward_type,
            #                  max_steps=max_steps, discount=self.gamma)
        self.state_dim = len(self.env.observation_space[0])
        self.action_dim = len(self.env.action_space)

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        print(f"Random Seed : {seed}")

        self.max_episodes = 2 * max_episodes  # TODO remove this length
        self.max_steps = max_steps
        self.max_frames = max_frames
        self.max_epochs = max_epochs
        self.update_timestep = self.max_steps * 4

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

        self.agent_str = algorithm

        print(f"Using the {self.agent_str} algorithm")

        self.chkpt_load = chkpt_load
        self.chkpt_load_path = chkpt_load_path
        self.load_multi = load_multi
        self.episode_count = 0

        if self.chkpt_load:
            if self.agent_str!= "MADDPG":
                self.episode_count = torch.load(self.chkpt_load_path)['episode_count']
            else:
                self.episode_count = 2000  # TODO change this hard-coding ep length
            self.max_episodes += self.episode_count

        if self.episode_count > 0:
            epsilon = 0.99
        else:
            epsilon = 1.0

        self.rational = rational
        self.rational_choice = rational_choice

        assert self.num_agents == len(self.rational), "Rationality definition does not match no. of agents"

        self.rational_ep = [True] * self.num_agents
        self.irrational_end = [0] * self.num_agents
        self.rational_const = 0.1  # tweak this param as well for rationality amount
        print(f"Rational behaviour: {self.rational}")
        if not all(self.rational):
            print(f"Rational choice: {self.rational_choice}")

        self.gradient_estimator = agent_algos.gradient_estimators.STGS(1.0)

        if self.agent_str == "DQN":
            self.agent = [DQN(self.state_dim, self.action_dim, epsilon=epsilon, rational_choice=self.rational_choice)
                          for _ in range(self.num_agents)]
        if self.agent_str == "PPO":
            self.agent = [PPO(self.state_dim,
                              self.action_dim,
                              lr_actor=3e-4,
                              lr_critic=3e-4,
                              gamma=0.95,
                              K_epochs=4,
                              eps_clip=0.2,
                              has_continuous_action_space=False) for _ in range(self.num_agents)]
        elif self.agent_str == "D3QN":
            self.agent = [D3QN(self.state_dim, self.action_dim, epsilon=epsilon, rational_choice=self.rational_choice)
                          for _ in range(self.num_agents)]
        elif self.agent_str == "MADDPG":
            self.agent = MADDPG(env=self.env,
                                critic_lr=3e-4,
                                actor_lr=3e-4,
                                gradient_clip=1.0,
                                hidden_dim_width=64,
                                gamma=0.95,
                                soft_update_size=0.01,
                                policy_regulariser=0.001,
                                gradient_estimator=self.gradient_estimator,
                                standardise_rewards=False,
                                pretrained_agents=None,
                                )
        else:
            sys.exit(0)

        self.test_actions = test_actions

    def plot_trajectory(self, colour, start_state=None, steps=800, fname=None, axes=None, fig=None):
        """To plot trajectories of the agent"""
        # state = self.env.reset_for_state(start_state)
        for episode in range(50):
            state_n, obs_n = self.env.reset()
            learning_progress = []
            actions = []
            rewards = []
            self.rational_ep = [True] * self.num_agents
            self.irrational_end = [0] * self.num_agents
            for step in range(steps):
                # list_state, list_reward_space = self.env.get_plot_state_list()

                for agent in range(self.num_agents):
                    self.rational_calc(agent, step)

                # take recommended action
                if self.agent_str == "MADDPG":
                    action_n = self.agent.acts(obs_n)
                else:
                    action_n = torch.tensor(
                        [self.agent[agent].get_action(obs_n[agent], rational=self.rational_ep[agent], testing=True) for
                         agent in
                         range(self.num_agents)])

                # Do the new chosen action in Environment
                next_state, reward, done, next_obs = self.env.step(action_n)
                actions.append(action_n)
                rewards.append(reward)
                learning_progress.append([state_n, action_n, reward])

                obs_n = next_obs.clone()
                state_n = next_state.clone()
                if torch.all(done):
                    break
            if self.agent_str == "MADDPG":
                fig, axes = self.env.plot_run(learning_progress, fig=fig, axes=None, fname='./' + str(episode) + '.png',
                                              colour=colour, maddpg=True)
            else:
                fig, axes = self.env.plot_run(learning_progress, fig=fig, axes=None, fname='./' + str(episode) + '.png',
                                              colour=colour)

        return actions, rewards

    def test_agent(self, n_points=100, max_steps=800, s_default=0.5):
        """Test the agent on different initial conditions to see if it can escape"""
        n_points = 900
        a_default = 0.5
        y_default = 0.5
        s_default = 0.5
        grid_size = int(np.sqrt(n_points))
        results = np.zeros((n_points, 1))
        # test_states = torch.zeros((n_points, self.num_agents, 3))
        test_states = torch.zeros((n_points, self.num_agents, 1))

        # y
        # lower_bound = 0.35
        # upper_bound = 0.65

        # epsilon
        # lower_bound = 160-40
        # upper_bound = 160+40

        # rho
        lower_bound = 2 - 1
        upper_bound = 2 + 1

        pointys = np.linspace(lower_bound, upper_bound, grid_size)

        for first in range(grid_size):
            for second in range(grid_size):
                # test_states[first * grid_size + second] = torch.tensor(  # Y variable
                #     [[a_default, pointys[first], s_default],
                #      [a_default, pointys[second], s_default]])
                # test_states[first * grid_size + second] = torch.tensor(  # S variable
                #     [[a_default, y_default, pointys[first]],
                #      [a_default, y_default, pointys[second]]])
                test_states[first * grid_size + second] = torch.tensor(  # epsilon and rho variable
                    [[pointys[first]], [pointys[second]]])

        for i in range(len(test_states)):
            # state_n, obs_n = self.env.reset(start_state=test_states.clone()[i])
            state_n, obs_n = self.env.reset(start_state=torch.tensor([[a_default, y_default, s_default],
                                                                      [a_default, y_default, s_default]]))

            # self.env.eps = test_states.clone()[i]  # epsilon
            self.env.rho = test_states.clone()[i]  # rho

            for steps in range(max_steps):
                if self.agent_str == "MADDPG":
                    action_n = self.agent.acts(obs_n)
                else:
                    action_n = torch.tensor(
                        [self.agent[agent].get_action(obs_n[agent], rational=self.rational_ep[agent], testing=True) for
                         agent in
                         range(self.num_agents)])
                next_state, reward, done, next_obs = self.env.step(action_n)
                if torch.all(done):
                    listy = []
                    for agent in range(self.num_agents):
                        listy.append(int(self.env.which_final_state(agent).value))

                    if any(x == 2 for x in listy):
                        results[i] = 1.0
                    else:
                        results[i] = 0.0

                    break
                obs_n = next_obs.clone()
                state_n = next_state.clone()

        return np.mean(results == 2), results, upper_bound, lower_bound

    def plot_end_state_matrix(self, results, upper_bound, lower_bound):
        size = int(np.sqrt(len(results)))
        fig = plt.imshow(np.flip(results.reshape(size, size), axis=0),
                         extent=(lower_bound, upper_bound, lower_bound, upper_bound), cmap='cividis')
        # plt.ylabel("Y Agent 0")
        # plt.xlabel("Y Agent 1")
        # plt.ylabel("S Agent 0")
        # plt.xlabel("S Agent 1")
        # plt.ylabel("Energy Efficiency Agent 0")
        # plt.xlabel("Energy Efficiency Agent 1")
        plt.ylabel("Rho Agent 0")
        plt.xlabel("Rho Agent 1")

        cbar = plt.colorbar(fig)
        cbar.set_label('Rate of Success')

        ax = plt.gca()

        ax.set_yticks([1.0, 1.5, 2.0, 2.5, 3.0])  # rho

        # ax.set_xticklabels([38, 47, 57, 70, 85, 105, 130])  # y
        # ax.set_yticklabels([38, 47, 57, 70, 85, 105, 130])

        # ax.set_xticklabels([270, 334, 410, 502, 613, 752, 931])  # s
        # ax.set_yticklabels([270, 334, 410, 502, 613, 752, 931])

        plt.tight_layout()
        # plt.show()
        save_name = "comparing_rho_agent_ranges"
        plt.savefig('./wandb_graphs/' + save_name + '.png', bbox_inches="tight", dpi=400)

    def append_data(self, episode_reward, episode):
        """We append the latest episode reward and calculate moving averages and moving standard deviations"""

        self.data['rewards'].append(episode_reward)

        concatenated = torch.stack(self.data['rewards'][-50:])

        mean = torch.mean(concatenated, dim=0)
        self.data['moving_avg_rewards'].append(mean.tolist())

        std = torch.std(concatenated, dim=0)
        self.data['moving_std_rewards'].append(std.tolist())

        self.data['episodes'] += 1

        if self.model == 'ays':
            final_states = [self.env.which_final_state(agent).name for agent in range(self.num_agents)]
            self.data['final_point'].append(final_states)

        # we log or print depending on settings
        if self.wandb_save:
            for agent in range(self.num_agents):
                label_1 = "train/episode_reward_agent_" + str(agent)
                label_2 = "train/moving_avg_rewards_agent_" + str(agent)
                # wandb.log({label_1: episode_reward[agent], "train/episode": episode})
                # wandb.log({label_2: self.data['moving_avg_rewards'][-1][agent][
                #     0], "train/episode": episode})
                wandb.log({label_1: episode_reward[agent], label_2: self.data['moving_avg_rewards'][-1][agent][
                    0], "train/episode": episode})

        if self.model == 'ays':
            print("Episode:", self.data['episodes'], "|| Reward:", round(episode_reward), "|| Final State ",
                  self.env.which_final_state().name) \
                if self.verbose else None

    def rational_calc(self, agent, step):
        if not self.rational[agent]:
            if self.rational_ep[agent]:
                if np.random.uniform() < self.rational_const:
                    self.rational_ep[agent] = False
                    self.irrational_end[agent] = step + np.random.randint(1, 10)  # adjust this range for irrationality
            else:
                if step == self.irrational_end[agent]:
                    self.rational_ep[agent] = True

    def pretrained_agents_load(self, algorithm):
        if self.chkpt_load:
            checkpoint = torch.load(self.chkpt_load_path)
            if algorithm == "MADDPG":
                self.agent = MADDPG(
                    env=self.env,
                    critic_lr=3e-4,
                    actor_lr=3e-4,
                    gradient_clip=1.0,
                    hidden_dim_width=64,
                    gamma=0.95,
                    soft_update_size=0.01,
                    policy_regulariser=0.001,
                    gradient_estimator=self.gradient_estimator,
                    standardise_rewards=False,
                    pretrained_agents=checkpoint,
                )
            elif self.agent_str == "PPO":
                for agent in range(self.num_agents):
                    chkpt_string = 'agent_' + str(agent)
                    (self.agent[agent].policy.load_state_dict(checkpoint[chkpt_string + '_state_dict']))
                self.episode_count = checkpoint['episode_count']
            else:
                for agent in range(self.num_agents):
                    chkpt_string = 'agent_' + str(agent)
                    self.agent[agent].target_net.load_state_dict(checkpoint[chkpt_string + '_target_state_dict'])
                    self.agent[agent].policy_net.load_state_dict(checkpoint[chkpt_string + '_policy_state_dict'])
                    self.agent[agent].optimizer.load_state_dict(checkpoint[chkpt_string + '_optimiser_state_dict'])
                self.episode_count = checkpoint['episode_count']

    def training_run(self):
        self.data['frame_idx'] = self.data['episodes'] = 0

        if self.wandb_save:
            # wandb.init(project="ucl_diss", group=self.model,
            wandb.init(project="ucl_diss", group="final_tests",
                       config={"reward_type": self.reward_type,
                               "observations": self.obs_type,
                               "rational": self.rational
                               })
            wandb.define_metric("train/episode")
            wandb.define_metric("train/*", step_metric="train/episode")

        if self.agent_str == "D3QN":
            memory = [utils.PER_IS_ReplayBuffer(self.buffer_size, alpha=self.alpha, state_dim=self.state_dim) for _ in
                      range(self.num_agents)]
        elif self.agent_str == "MADDPG":
            memory = utils.MADDPG_ReplayBuffer(2_000_000, [self.state_dim] * self.num_agents, 512)
        else:
            memory = [utils.ReplayBuffer(self.buffer_size) for _ in range(self.num_agents)]

        time_step = 0

        for episodes in range(self.max_episodes):
            state_n, obs_n = self.env.reset()
            self.early_finish = [False] * self.num_agents
            episode_reward = torch.tensor([0.0]).repeat(self.num_agents, 1)

            self.rational_ep = [True] * self.num_agents
            self.irrational_end = [0] * self.num_agents

            if self.animation:
                x_values = [[] for _ in range(self.num_agents)]
                y_values = [[] for _ in range(self.num_agents)]
                z_values = [[] for _ in range(self.num_agents)]
                reward_values = [[] for _ in range(self.num_agents)]

            for i in range(self.max_steps):

                for agent in range(self.num_agents):
                    self.rational_calc(agent, i)

                if self.agent_str == "MADDPG":
                    action_n = self.agent.acts(obs_n)
                else:
                    action_n = torch.tensor(
                        [self.agent[agent].get_action(obs_n[agent], rational=self.rational_ep[agent]) for agent in
                         range(self.num_agents)])
                if self.test_actions:
                    action_n = torch.tensor([0 for _ in range(self.num_agents)])

                for agent in range(self.num_agents):
                    label = "action_n_agent_" + str(agent)
                    wandb.log({label: action_n[agent]}) if self.wandb_save else None

                # print(action_n)
                # sys.exit()

                # step through environment
                next_state, reward, done, next_obs = self.env.step(action_n)

                if self.agent_str == "MADDPG":
                    memory.store(
                        obs=obs_n,
                        acts=action_n,
                        rwds=reward.view(2),
                        nobs=next_obs,
                        dones=done.view(2),
                    )
                    for agent in range(self.num_agents):
                        episode_reward[agent] += reward[agent]
                        if done[agent]:
                            self.early_finish[agent] = True
                elif self.agent_str == "PPO":
                    time_step += 1
                    for agent in range(self.num_agents):
                        self.agent[agent].buffer.rewards.append(reward[agent])
                        self.agent[agent].buffer.is_terminals.append(done[agent])
                        if time_step % self.update_timestep == 0:
                            self.agent[agent].update()
                        if done[agent]:
                            self.early_finish[agent] = True
                else:
                    for agent in range(self.num_agents):
                        # if not self.early_finish[agent]:
                        episode_reward[agent] += reward[agent]
                        memory[agent].push(obs_n[agent], action_n[agent], reward[agent], next_obs[agent], done[agent])
                        if len(memory[agent]) > self.batch_size:
                            if self.agent_str == "D3QN":
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

                            label = "train/loss_agent_" + str(agent)
                            wandb.log({label: loss}) if self.wandb_save else None
                        if done[agent]:
                            self.early_finish[agent] = True
                    # else:  # allows another agent to carry on running even if one has reached final state
                    #     next_state[agent] = state_n[agent].clone()

                obs_n = next_obs.clone()
                state_n = next_state.clone()
                reward_clone = reward.clone()  # TODO double check this is correct for the rest
                self.data['frame_idx'] += 1

                if self.animation:
                    for agent in range(self.num_agents):
                        x_values[agent].append(state_n[agent][0])
                        y_values[agent].append(state_n[agent][1])
                        z_values[agent].append(state_n[agent][2])
                        reward_values[agent].append(reward_clone[agent][0])

                if torch.all(done):
                    break

            def animate_func(num):
                for line in ax3d.lines:  # TODO why does this start with two empty lists?
                    line.remove()

                for agent in range(self.num_agents):
                    if self.early_finish[agent]:
                        ax3d.plot3D(xs=x_values[agent][:num], ys=y_values[agent][:num], zs=z_values[agent][:num],
                                    color='r', alpha=0.8, lw=3, label=f"Agent : {agent}")
                    else:
                        ax3d.plot3D(xs=x_values[agent][:num], ys=y_values[agent][:num], zs=z_values[agent][:num],
                                    color=colors[agent], alpha=0.8, lw=3, label=f"Agent : {agent}")

                ax3d.set_title(
                    [f"{self.agent_str} Agent {agent} {self.reward_type[agent]} reward : {reward_values[agent][num]:.2f}" for agent in
                     range(self.num_agents)])

            if self.animation and episodes % 100 == 0:
                if self.model == 'ays':
                    fig, ax3d = create_figure_ays(top_down=self.top_down)
                if self.model == 'rice-n':
                    fig, ax3d = create_figure_ricen(top_down=self.top_down)
                colors = plt.cm.brg(np.linspace(0, 1, self.num_agents))
                line_ani = FuncAnimation(fig, animate_func, interval=5, frames=len(x_values[0]), repeat=False)
                file_name = f"gifs/{episodes}.gif"
                line_ani.save(file_name, writer=PillowWriter(fps=60))
                print(f"Saved GIF of episode - {episodes}")

            self.append_data(episode_reward, episodes)

            if self.agent_str == "MADDPG":
                sample = memory.sample()
                if sample is not None:
                    self.agent.update(sample)

        self.episode_count += self.max_episodes

        # add checkpointing here
        chkpt_save_path = './checkpoints/env=' + str(self.model) + '_reward_type=' + str(self.reward_type) \
                          + '_obs_type=' + str(self.obs_type) + '_num_agents=' + str(self.num_agents) \
                          + '_homogeneous=' + str(self.homogeneous) + '_rationality=' + str(self.rational) \
                          + '_rational_choice=' + str(self.rational_choice) + '_episodes=' + str(self.episode_count)
        tot_dict = {}
        if self.agent_str == "MADDPG":
            chkpt_save_path += '_MADDPG'
            if not os.path.exists(chkpt_save_path):
                os.makedirs(chkpt_save_path)
            torch.save(self.agent.agents,
                       chkpt_save_path + '/end_time=' + str(datetime.now().strftime("%d-%m-%Y_%H-%M-%S")) + '.tar')
        elif self.agent_str == "PPO":
            chkpt_save_path += '_PPO'
            if not os.path.exists(chkpt_save_path):
                os.makedirs(chkpt_save_path)
            for agent in range(self.num_agents):
                tot_dict['agent_' + str(agent) + '_state_dict'] = self.agent[agent].policy.state_dict()
            tot_dict['episode_count'] = self.episode_count
            torch.save(tot_dict,
                       chkpt_save_path + '/end_time=' + str(datetime.now().strftime("%d-%m-%Y_%H-%M-%S")) + '.tar')
        else:
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

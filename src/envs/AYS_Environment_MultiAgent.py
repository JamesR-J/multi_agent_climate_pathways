"""
This is the implementation of the AYS Environment in the form
that it can used within the Agent-Environment interface
in combination with the DRL-agent.

@author: Felix Strnad, Theodore Wolf

"""

import sys
import torch
import os

import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import odeint
from gym import Env
from enum import Enum
from inspect import currentframe, getframeinfo

from .graph_functions import create_figure_ays
from . import graph_functions as ays_plot, ays_model as ays


class Basins(Enum):
    OUT_PB = 0
    BLACK_FP = 1
    GREEN_FP = 2

    A_PB = 3
    Y_SF = 4
    E_PB = 5

    OUT_OF_TIME = 6


class AYS_Environment(Env):
    """
    The environment is based on Kittel et al. 2017, and contains in part code adapted from
    https://github.com/timkittel/ays-model/ .
    This Environment describes the 3D implementation of a simple model for the development of climate change, wealth
    and energy transformation which is inspired by the model from Kellie-Smith and Cox.
    Dynamic variables are :
        - excess atmospheric carbon stock A
        - the economic output/production Y  (similar to wealth)
        - the renewable energy knowledge stock S

    Parameters
    ----------
         - sim_time: Timestep that will be integrated in this simulation step
          In each grid point the agent can choose between subsidy None, A, B or A and B in combination.
    """
    dimensions = np.array(['A', 'Y', 'S'])
    management_options = ['default', 'LG', 'ET', 'LG+ET']
    action_space = torch.tensor([[False, False], [True, False], [False, True], [True, True]])
    action_space_number = np.arange(len(action_space))
    # AYS example from Kittel et al. 2017:
    tau_A = 50  # carbon decay - single val
    tau_S = 50  # renewable knowledge stock decay - probs single val
    beta = 0.03  # economic output growth - multi val
    beta_LG = 0.015  # halved economic output growth - multi val
    eps = 147  # energy efficiency param - single val
    A_offset = 600  # i have no idea # TODO check this
    theta = beta / (950 - A_offset)  # beta / ( 950 - A_offset(=350) )
    # theta = 8.57e-5

    rho = 2.  # renewable knowledge learning rate - single val?
    sigma = 4e12  # break even knowledge - multi val
    sigma_ET = sigma * 0.5 ** (1 / rho)  # can't remember the change, but it's somewhere - multi val
    # sigma_ET = 2.83e12

    phi = 4.7e10

    # AYS0 = [240, 7e13, 5e11]

    possible_test_cases = [[0.4949063922255394, 0.4859623171738628, 0.5], [0.42610779, 0.52056811, 0.5]]

    def __init__(self, gamma=0.99, t0=0, dt=1, reward_type='PB', max_steps=600, image_dir='./images/', run_number=0,
                 plot_progress=False, num_agents=2, obs_type='all_agents'):
        self.management_cost = 0.5
        self.image_dir = image_dir
        self.run_number = run_number
        self.plot_progress = plot_progress
        self.max_steps = max_steps
        self.gamma = gamma

        self.num_agents = num_agents
        self.tau_A = torch.tensor([self.tau_A]).repeat(self.num_agents, 1)
        self.tau_S = torch.tensor([self.tau_S]).repeat(self.num_agents, 1)
        self.beta = torch.tensor([self.beta]).repeat(self.num_agents, 1)
        self.beta_LG = torch.tensor([self.beta_LG]).repeat(self.num_agents, 1)
        self.eps = torch.tensor([self.eps]).repeat(self.num_agents, 1)
        self.theta = torch.tensor([self.theta]).repeat(self.num_agents, 1)
        self.rho = torch.tensor([self.rho]).repeat(self.num_agents, 1)
        self.sigma = torch.tensor([self.sigma]).repeat(self.num_agents, 1)
        self.sigma_ET = torch.tensor([self.sigma_ET]).repeat(self.num_agents, 1)
        self.phi = torch.tensor([self.phi]).repeat(self.num_agents, 1)

        # The grid defines the number of cells, hence we have 8x8 possible states
        self.final_state = torch.tensor([False]).repeat(self.num_agents, 1)
        self.reward = torch.tensor([0.0]).repeat(self.num_agents, 1)
        self.reward_type = reward_type
        print("Reward type: {}".format(self.reward_type))

        self.timeStart = 0
        self.intSteps = 10  # integration Steps
        self.t = self.t0 = t0
        self.dt = dt

        self.sim_time_step = np.linspace(self.timeStart, dt, self.intSteps)

        self.green_fp = torch.tensor([0, 1, 0]).repeat(self.num_agents, 1)
        self.brown_fp = torch.tensor([1, 0.4, 0.6]).repeat(self.num_agents, 1)
        self.final_radius = torch.tensor([0.05]).repeat(self.num_agents, 1)
        self.color_list = ays_plot.color_list

        self.X_MID = [240, 7e13, 501.5198]

        self.state = self.current_state = torch.tensor([0.5, 0.5, 0.5]).repeat(self.num_agents, 1)
        self.reward_space = torch.tensor([0.5, 0.5, 0.5, 10.0 / 1003.04]).repeat(self.num_agents, 1)
        self.obs_type = obs_type
        print("Observation type: {}".format(self.obs_type))
        if self.obs_type == 'agent_only':
            self.observation_space = torch.tensor([0.5, 0.5, 0.5, 10.0 / 1003.04]).repeat(self.num_agents, 1)
        elif self.obs_type == 'all_shared':
            self.observation_space = torch.cat((torch.eye(self.num_agents), torch.tensor([0.5]).repeat(self.num_agents, 1), torch.tensor([0.5, 0.5, 10.0] * self.num_agents).repeat(self.num_agents, 1)), dim=1)

        """
        This values define the planetary boundaries of the AYS model
        """
        self.A_PB = torch.tensor([self._compactification(ays.boundary_parameters["A_PB"], self.X_MID[0])]).repeat(self.num_agents, 1)  # Planetary boundary: 0.5897
        self.Y_SF = torch.tensor([self._compactification(ays.boundary_parameters["W_SF"], self.X_MID[1])]).repeat(self.num_agents, 1)  # Social foundations as boundary: 0.3636
        self.E_LIMIT = torch.tensor([1.0]).repeat(self.num_agents, 1)
        self.PB = torch.cat((self.E_LIMIT, self.Y_SF, self.A_PB), dim=1)
        self.PB_2 = torch.cat((self.A_PB, self.Y_SF, torch.tensor([0.0]).repeat(self.num_agents, 1)), dim=1)
        self.PB_3 = torch.cat((torch.tensor([0.0]).repeat(self.num_agents, 1), self.Y_SF, torch.tensor([0.0]).repeat(self.num_agents, 1), torch.tensor([0.0]).repeat(self.num_agents, 1)), dim=1)

    def step(self, action: int):

        next_t = self.t + self.dt

        result = self._perform_step(action, next_t)  # A Y S E
        self.state[:, 0] = result[:, 3]
        self.state[:, 1] = result[:, 1]
        self.state[:, 2] = result[:, 0]
        self.observation_space, self.reward_space = self.generate_observation(result)

        if not self.final_state.bool().any():
            assert torch.all(self.state[:, 2] == self.state[0, 2]), "Values in the A column are not all equal"

        self.t = next_t

        self.get_reward_function()  # TODO check if this might be needed before step is done to evaluate the current state, not the next state!

        for agent in range(self.num_agents):
            if self._arrived_at_final_state(agent):
                self.final_state[agent] = True
            # if not self._inside_A_pb(agent):  # TODO maybe adjust this so insideA_PB used for sigle agent and the other for both? but then if both agents outside the y pb then the current thing would end but this one would not, hmmmm maybe test it ya know
            if not self._inside_planetary_boundaries(agent):
                self.final_state[agent] = True

        if torch.all(self.final_state):
            for agent in range(self.num_agents):
                # if self.reward_type[agent] == "PB" or self.reward_type[agent] == "policy_cost":  # TODO how will this work with different reward functions
                self.reward[agent] += self.calculate_expected_final_reward(agent)
        else:
            self.final_state = torch.tensor([False]).repeat(self.num_agents, 1)

        # print(self.reward)
        # print(self.state)

        return self.state, self.reward, self.final_state, self.observation_space

    def generate_observation(self, ode_int_output):
        if self.obs_type == "agent_only":
            return ode_int_output, ode_int_output

        elif self.obs_type == "all_shared":
            result = ode_int_output[:, 1:].flatten().repeat(self.num_agents, 1)
            result = torch.cat((torch.eye(self.num_agents), ode_int_output[:, 0].view(self.num_agents, 1), result), dim=1)
            return result, ode_int_output

    def _perform_step(self, action, next_t):

        parameter_matrix = self._get_parameters(action)

        parameter_vector = parameter_matrix.flatten()
        parameter_vector = torch.cat((parameter_vector, torch.tensor([self.num_agents])))

        # if self.obs_type == "agent_only":  # TODO think can remove this with the new self.reward_space
        #     ode_input = torch.cat((self.observation_space[:, 0:3], torch.zeros((self.num_agents, 1))), dim=1)
        # elif self.obs_type == "all_shared":
        #     input = torch.cat((self.observation_space[0, self.num_agents].repeat(self.num_agents, 1), self.observation_space[0, (self.num_agents + 1):].view(self.num_agents, 3)[:, :2]), dim=1)
        #     ode_input = torch.cat((input, torch.zeros((self.num_agents, 1))), dim=1)

        ode_input = torch.cat((self.reward_space[:, 0:3], torch.zeros((self.num_agents, 1))), dim=1)

        traj_one_step = odeint(ays.AYS_rescaled_rhs_marl2, ode_input.flatten(), [self.t, next_t], args=tuple(parameter_vector.tolist()), mxstep=50000)

        return torch.tensor(traj_one_step[1]).view(-1, 4)  # A Y S E output

    def reset(self):
        # self.state=np.array(self.random_StartPoint())
        self.state = self.current_state_region_StartPoint()

        self.final_state = torch.tensor([False]).repeat(self.num_agents, 1)
        self.t = self.t0  # TODO reinstate reset for observation_space

        self.reward_space = torch.tensor([0.5, 0.5, 0.5, 10.0 / 1003.04]).repeat(self.num_agents, 1)

        if self.obs_type == 'agent_only':
            self.observation_space = torch.cat((self.state, torch.tensor(10.0 / 1003.04).repeat(self.num_agents, 1)), dim=1)  # TODO dodgy fix assuming emissions at 10 - need to fix this really
        elif self.obs_type == 'all_shared':
            mid = torch.cat((self.state[:, 1:], torch.tensor([10.0 / 1003.04]).repeat(self.num_agents, 1)), dim=1).flatten().repeat(self.num_agents, 1)
            self.observation_space = torch.cat((torch.eye(self.num_agents), self.state[:, 0].view(self.num_agents, 1), mid), dim=1)

        return self.state, self.observation_space

    def get_reward_function(self):
        def policy_cost(agent, action=0):  # TODO check this if works
            """@Theo Wolf, we add a cost to using management options """
            if self._inside_planetary_boundaries(agent):
                self.reward[agent] = torch.norm(self.state[agent] - self.PB[agent])
            else:
                self.reward[agent] = 0.0  # TODO make a version with the -100 for hitting planetary boundary - or make it so like remaining steps each add a -1 or something as it outside the boundary etc - will figure out best way to do it ya know
            if self.management_options[action] != 'default':
                #reward -= self.management_cost  # cost of using management
                self.reward[agent] *= self.management_cost
                if self.management_options[action] == 'LG+ET':
                    # reward -= self.management_cost  # we add more cost for using both actions
                    self.reward[agent] *= self.management_cost

        def reward_distance_PB(agent, action=0):
            self.reward[agent] = 0.0

            if self._inside_planetary_boundaries(agent):
                # self.reward[agent] = torch.norm(self.state[agent]-self.PB[agent])
                # self.reward[agent] = torch.norm(self.state[agent, 1:] - self.PB[agent, 1:])
                self.reward[agent] = torch.norm(self.reward_space[agent, :3] - self.PB_2[agent])  # TODO check works fine with reward_space instead
            else:
                self.reward[agent] = 0.0  # -torch.norm(self.state[agent]-self.PB[agent])  # TODO maybe look at some negative thing the further outside the ideal space it is

        def reward_distance_PB_new(agent, action=0):
            self.reward[agent] = 0.0

            if self._inside_planetary_boundaries(agent):
                self.reward[agent] = torch.norm(self.state[agent]-self.PB[agent])
            else:
                # print(self._which_PB(agent))
                if self.state[agent, 0] >= self.A_PB[agent]:   # TODO implemented bad reward for hitting PB is that good idea idk - needs fixing with latest state updates too
                    self.reward[agent] += -100
                if self.state[agent, 1] <= self.Y_SF[agent]:
                    self.reward[agent] += -100

        def reward_distance_Y(agent, action=0):
            self.reward[agent] = 0.

            self.reward[agent] = torch.norm(self.reward_space[agent, :2] - self.PB_3[agent, :2])  # max a and y

            # if self._inside_planetary_boundaries(agent):  # TODO do we need to check even in PB as we don't care about PBs
            #     self.reward[agent] = torch.norm(self.reward_space[agent, :2] - self.PB_3[agent, :2])
            # else:
            #     self.reward[agent] = 0.0

        def reward_distance_E(agent, action=0):
            self.reward[agent] = 0.

            self.reward[agent] = torch.abs(self.reward_space[agent, 3] - self.PB_3[agent, 3])  # max e

        for agent in range(self.num_agents):
            if self.reward_type[agent] == 'PB':
                reward_distance_PB(agent)
            elif self.reward_type[agent] == 'PB_new':
                reward_distance_PB_new(agent)
            elif self.reward_type[agent] == 'max_Y':
                reward_distance_Y(agent)
            elif self.reward_type[agent] == 'max_E':
                reward_distance_E(agent)
            elif self.reward_type[agent] == "policy_cost":
                policy_cost(agent)
            elif self.reward_type[agent] == None:
                print("ERROR! You have to choose a reward function!\n",
                      "Available Reward functions for this environment are: PB, rel_share, survive, desirable_region!")
                exit(1)
            else:
                print("ERROR! The reward function you chose is not available! " + self.reward_type[agent])
                self.print_debug_info()
                sys.exit(1)

    def calculate_expected_final_reward(self, agent):
        """
        Get the reward in the last state, expecting from now on always default.
        This is important since we break up simulation at final state, but we do not want the agent to
        find trajectories that are close (!) to final state and stay there, since this would
        result in a higher total reward.
        """
        remaining_steps = self.max_steps - self.t
        discounted_future_reward = 0.
        for i in range(remaining_steps):
            discounted_future_reward += self.gamma ** i * self.reward[agent]

        # print("Agent number : {}".format(agent))
        # print(discounted_future_reward)
        return discounted_future_reward

    @staticmethod
    def print_debug_info():
        frameinfo = getframeinfo(currentframe())
        print("File: ", frameinfo.filename)

    @staticmethod
    def _compactification(x, x_mid):
        if x == 0:
            return 0.
        if x == np.infty:
            return 1.
        return x / (x + x_mid)

    @staticmethod
    def _inv_compactification(y, x_mid):
        if y == 0:
            return 0.
        if np.allclose(y, 1):
            return np.infty
        return x_mid * y / (1 - y)

    def _inside_A_pb(self, agent):  # TODO think this needed for single agent stuff
        e = self.state[agent, 0]
        y = self.state[agent, 1]
        a = self.state[agent, 2]
        is_inside = True

        if a > self.A_PB[agent]:
            is_inside = False
            # print("Outside PB!")
        return is_inside

    def _inside_planetary_boundaries(self, agent):  # TODO confirm this is correct
        e = self.state[agent, 0]
        y = self.state[agent, 1]
        a = self.state[agent, 2]
        is_inside = True

        if a > self.A_PB[agent] or y < self.Y_SF[agent] or e > self.E_LIMIT[agent]:
            is_inside = False
            # print("Outside PB!")
        return is_inside

    def _inside_planetary_boundaries_all(self):  # TODO confirm this is correct
        e = self.state[:, 0]
        y = self.state[:, 1]
        a = self.state[:, 2]
        is_inside = True

        if torch.all(a > self.A_PB) or torch.all(y < self.Y_SF) or torch.all(e > self.E_LIMIT):
            is_inside = False
            # print("Outside PB!")
        return is_inside

    def _arrived_at_final_state(self, agent):
        e = self.state[agent, 0]
        y = self.state[agent, 1]
        a = self.state[agent, 2]

        if torch.abs(e - self.green_fp[agent, 0]) < self.final_radius[agent] \
                and torch.abs(y - self.green_fp[agent, 1]) < self.final_radius[agent]\
                and torch.abs(a - self.green_fp[agent, 2]) < self.final_radius[agent]:
            return True
        elif torch.abs(e - self.brown_fp[agent, 0]) < self.final_radius[agent]\
                and torch.abs(y - self.brown_fp[agent, 1]) < self.final_radius[agent]\
                and torch.abs(a - self.brown_fp[agent, 2]) < self.final_radius[agent]:
            return True
        else:
            return False

    def _good_final_state(self, agent):  # TODO confirm this is correct
        e = self.state[agent, 0]
        y = self.state[agent, 1]
        a = self.state[agent, 2]
        if np.abs(e - self.green_fp[agent, 0]) < self.final_radius[agent]\
                and np.abs(y - self.green_fp[agent, 1]) < self.final_radius[agent]\
                and np.abs(a - self.green_fp[agent, 2]) < self.final_radius[agent]:
            return True
        else:
            return False

    def which_final_state(self, agent):  # TODO confirm this is correct
        e = self.state[agent, 0]
        y = self.state[agent, 1]
        a = self.state[agent, 2]
        if np.abs(a - self.green_fp[agent, 0]) < self.final_radius[agent] and np.abs(y - self.green_fp[agent, 1]) < self.final_radius[agent] and np.abs(a - self.green_fp[agent, 2]) < self.final_radius[agent]:
            # print("ARRIVED AT GREEN FINAL STATE WITHOUT VIOLATING PB!")
            return Basins.GREEN_FP
        elif np.abs(a - self.brown_fp[agent,  0]) < self.final_radius[agent] and np.abs(y - self.brown_fp[agent, 1]) < self.final_radius[agent] and np.abs(a - self.brown_fp[agent, 2]) < self.final_radius[agent]:
            return Basins.BLACK_FP
        else:
            # return Basins.OUT_PB
            return self._which_PB(agent)

    def _which_PB(self, agent):  # TODO confirm this is correct
        """ To check which PB has been violated"""
        if self.state[agent, 2] >= self.A_PB[agent]:
            return Basins.A_PB
        elif self.state[agent, 1] <= self.Y_SF[agent]:
            return Basins.Y_SF
        elif self.state[agent, 0] <= 0:
            return Basins.E_PB
        else:
            return Basins.OUT_OF_TIME

    def random_StartPoint(self):

        self.state = torch.tensor([0, 0, 0]).repeat(self.num_agents, 1)
        while not self._inside_planetary_boundaries_all():
            self.state = torch.tensor(np.random.uniform(size=(self.current_state.size(0), 2)))

        return self.state

    def current_state_region_StartPoint(self):

        self.state = torch.tensor([0, 0, 0]).repeat(self.num_agents, 1)
        limit_start = 0.05

        while not self._inside_planetary_boundaries_all():

            adjustment = torch.tensor(
                np.random.uniform(low=-limit_start, high=limit_start, size=(self.current_state.size(0), 2)))
            self.state = self.current_state.clone()
            self.state[:, :2] += adjustment

            const_val = self.state[0, 0]
            self.state[:, 0] = const_val

            # self.state[:, 2] = 0.2  # TODO just trying to force it lower idk but this does not work defo

            # print(self.state)

            assert torch.allclose(self.state[:, 0], const_val), "First column values are not equal."

        return self.state

    def _inside_box(self):
        """
        This function is needed to check whether our system leaves the predefined box (1,1,1).
        If values turn out to be negative, this is physically false, and we stop simulation and treat as a final state.
        """
        inside_box = True
        for x in self.state:
            if x < 0:
                x = 0
                inside_box = False
        return inside_box

    def _get_parameters(self, action=None):

        """
        This function is needed to return the parameter set for the chosen management option.
        Here the action numbers are really transformed to parameter lists, according to the chosen
        management option.
        Parameters:
            -action_number: Number of the action in the actionset.
             Can be transformed into: 'default', 'degrowth' ,'energy-transformation' or both DG and ET at the same time
        """
        # if action < len(self.action_space):
        #     action_tuple = self.action_space[action]
        # else:
        #     print("ERROR! Management option is not available!" + str(action))
        #     print(get_linenumber())
        #     sys.exit(1)

        if action is None:
            action = torch.tensor([0]).repeat(self.num_agents, 1)

        # if type(action) == int:
        #     action = torch.tensor([action]).view(self.num_agents, 1)

        selected_rows = self.action_space[action.squeeze(), :]
        action_matrix = selected_rows.view(self.num_agents, 2)

        mask_1 = action_matrix[:, 0].unsqueeze(1)
        mask_2 = action_matrix[:, 1].unsqueeze(1)

        beta = torch.where(mask_1, self.beta_LG, self.beta)
        sigma = torch.where(mask_2, self.sigma_ET, self.sigma)

        parameter_matrix = torch.cat((beta, self.eps, self.phi, self.rho, sigma, self.tau_A, self.tau_S, self.theta), dim=1)

        return parameter_matrix

    # def plot_run(self, learning_progress, fig, axes, colour, fname=None,):
    #     timeStart = 0
    #     intSteps = 2  # integration Steps
    #     dt = 1
    #     sim_time_step = np.linspace(timeStart, self.dt, intSteps)
    #     if axes is None:
    #         fig, ax3d = create_figure_ays()
    #     else:
    #         ax3d = axes
    #     start_state = learning_progress[0][0]
    #
    #     for state_action in learning_progress:
    #         state = state_action[0]
    #         action = state_action[1]
    #         parameter_list = self._get_parameters(action)
    #         traj_one_step = odeint(ays.AYS_rescaled_rhs, state, sim_time_step, args=parameter_list[0])
    #         # Plot trajectory
    #         my_color = ays_plot.color_list[action]
    #         ax3d.plot3D(xs=traj_one_step[:, 0], ys=traj_one_step[:, 1], zs=traj_one_step[:, 2],
    #                     color=colour, alpha=0.3, lw=3)
    #
    #     # Plot from startpoint only one management option to see if green fix point is easy to reach:
    #     # self.plot_current_state_trajectories(ax3d)
    #     ays_plot.plot_hairy_lines(20, ax3d)
    #
    #     final_state = self.which_final_state().name
    #     if fname is not None:
    #         plt.savefig(fname)
    #     #plt.show()
    #
    #     return fig, ax3d

    # def observed_states(self):
    #     return self.dimensions

    # def plot_current_state_trajectories(self, ax3d):
    #     # Trajectories for the current state with all possible management options
    #     time = np.linspace(0, 300, 1000)
    #
    #     for action_number in range(len(self.action_space)):
    #         parameter_list = self._get_parameters(action_number)
    #         my_color = self.color_list[action_number]
    #         traj_one_step = odeint(ays.AYS_rescaled_rhs, self.current_state, time, args=parameter_list[0])
    #         ax3d.plot3D(xs=traj_one_step[:, 0], ys=traj_one_step[:, 1], zs=traj_one_step[:, 2],
    #                     color=my_color, alpha=.7, label=None)

    # def save_traj_final_state(self, learners_path, file_path, episode):
    #     final_state = self.which_final_state().name
    #
    #     states = np.array(learners_path)[:, 0]
    #     start_state = states[0]
    #     a_states = list(zip(*states))[0]
    #     y_states = list(zip(*states))[1]
    #     s_states = list(zip(*states))[2]
    #
    #     actions = np.array(learners_path)[:, 1]
    #     rewards = np.array(learners_path)[:, 2]
    #
    #     full_file_path = file_path + '/DQN_Path/' + final_state + '/'
    #     if not os.path.isdir(full_file_path):
    #         os.makedirs(full_file_path)
    #
    #     text_path = (full_file_path +
    #                  str(self.run_number) + '_' + 'path_' + str(start_state) + '_episode' + str(episode) + '.txt')
    #     with open(text_path, 'w') as f:
    #         f.write("# A  Y  S   Action   Reward \n")
    #         for i in range(len(learners_path)):
    #             f.write("%s  %s  %s   %s   %s \n" % (a_states[i], y_states[i], s_states[i], actions[i], rewards[i]))
    #     f.close()
    #     print('Saved :' + text_path)
    #
    # def save_traj(self, ax3d, fn):
    #     ax3d.legend(loc='best', prop={'size': 12})
    #     plt.savefig(fname=fn)
    #     plt.close()

    def define_test_points(self):
        testpoints = [
            [0.49711988, 0.49849855, 0.5],
            [0.48654806, 0.51625583, 0.5],
            [0.48158348, 0.50938806, 0.5],
            [0.51743486, 0.45828958, 0.5],
            [0.52277734, 0.49468274, 0.5],
            [0.49387675, 0.48199759, 0.5],
            [0.45762969, 0.50656114, 0.5]
        ]
        return testpoints

    def test_Q_states(self):
        # The Q values are choosen here in the region of the knick and the corner
        testpoints = [
            [0.5, 0.5, 0.5],
            [0.48158348, 0.50938806, 0.5],  # points around current state
            [0.51743486, 0.45828958, 0.5],
            [0.52277734, 0.49468274, 0.5],
            [0.49711988, 0.49849855, 0.5],
            [0.5642881652513302, 0.4475774101441196, 0.5494879542441825],  # From here on for knick to green FP
            [0.5677565382994565, 0.4388184256945361, 0.5553589418072845],
            [0.5642881652513302, 0.4475774101441196, 0.5494879542441825],
            [0.5667064632786063, 0.4417642808582638, 0.5534355600174762],
            [0.5677565382994565, 0.4388184256945361, 0.5553589418072845],
            [0.5667064632786063, 0.4417642808582638, 0.5534355600174762],
            [0.5642881652513302, 0.4475774101441196, 0.5494879542441825],
            [0.5667064632786063, 0.4417642808582638, 0.5534355600174762],
            [0.5677565382994565, 0.4388184256945361, 0.5553589418072845],
            [0.5667064632786063, 0.4417642808582638, 0.5534355600174762],
            [0.565551647191721, 0.4446849282686741, 0.5514780427327116],
            [0.5667064632786063, 0.4417642808582638, 0.5534355600174762],
            [0.5732889740892303, 0.40670386098365746, 0.5555233190964499],
            [0.575824650184652, 0.4053645419804867, 0.4723020776953208],
            [0.5770448313058577, 0.4048031241155815, 0.418890921031026],  # From here on for knick to black FP
            [0.5731695199856403, 0.40703303828389187, 0.5611291038925613],
            [0.5742215704891825, 0.42075928220225944, 0.4638131691273601],
            [0.5763299679962532, 0.411095026888074, 0.4294020150808698],
            [0.5722546035810613, 0.41315124675768045, 0.5695919593600399],
            [0.5762062083990029, 0.405168276738863, 0.4567816125395152],
            [0.5762327254875753, 0.4052313013623205, 0.4568789522146076],
            [0.5770448313058577, 0.4048031241155815, 0.418890921031026],
            [0.5770448313058577, 0.4048031241155815, 0.418890921031026],
            [0.5726685871808355, 0.40709323935138103, 0.5727121746516005],
            [0.2841645298525685, 0.5742868996790442, 0.9699317116062534],  # From here on region of the shelter
            [0.32909951420599637, 0.6082136751752725, 0.9751810127843358],
            [0.5649255262907135, 0.4238116683903446, 0.8009508342049909],
            [0.04143141196994614, 0.9467759116676885, 0.9972458138530155],
        ]
        return testpoints

    def test_reward_functions(self):
        print(self.reward_type)
        print(self.state)
        self.state[0, 0] = 0.5899
        self.state[0, 1] = 0.362
        print(self.state)
        print(self.reward)
        self.get_reward_function()
        print(self.reward)
        sys.exit()











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


from gym import Env
from enum import Enum
from inspect import currentframe, getframeinfo

from .graph_functions import create_figure_ays
from . import graph_functions as ays_plot, ays_model as ays
from .ays_model import AYS_rescaled_rhs_marl2
from flax import struct
import chex
import jax
import jax.numpy as jnp
import jax.random as jrandom
from functools import partial
from typing import Tuple, Dict
from jaxmarl.environments.spaces import Discrete, MultiDiscrete
from jax.experimental.ode import odeint


# class Basins(Enum):
#     OUT_PB = 0
#     BLACK_FP = 1
#     GREEN_FP = 2
#
#     A_PB = 3
#     Y_SF = 4
#     E_PB = 5
#
#     OUT_OF_TIME = 6


@struct.dataclass
class EnvState:  # TODO fill this env state up as we need it basos
    ayse: chex.Array
    terminal: bool
    step: int


class AYS_Environment(object):
    def __init__(self, gamma=0.99, t0=0, dt=1, reward_type='PB', max_steps=600, image_dir='./images/', run_number=0,
                 plot_progress=False, num_agents=3):
        self.management_cost = 0.5
        self.image_dir = image_dir
        self.run_number = run_number
        self.plot_progress = plot_progress
        self.max_steps = max_steps
        self.gamma = gamma

        self.num_agents = num_agents
        self.agents = [f"agent_{i}" for i in range(num_agents)]

        # The grid defines the number of cells, hence we have 8x8 possible states
        self.final_state = torch.tensor([False]).repeat(self.num_agents, 1)
        self.reward = torch.tensor([0.0]).repeat(self.num_agents, 1)
        self.reward_type = reward_type
        print(f"Reward type: {self.reward_type}")

        self.timeStart = 0
        self.intSteps = 10  # integration Steps
        self.t = self.t0 = t0
        self.dt = dt
        self.sim_time_step = np.linspace(self.timeStart, dt, self.intSteps)

        self.green_fp = jnp.tile([0, 1, 0], self.num_agents)  # TODO sort these out babY
        print(self.green_fp)
        sys.exit()
        self.brown_fp = torch.tensor([1, 0.4, 0.6]).repeat(self.num_agents, 1)
        self.final_radius = torch.tensor([0.05]).repeat(self.num_agents, 1)
        self.color_list = ays_plot.color_list

        self.X_MID = [240, 7e13, 501.5198]

        self.game_actions = {"NOTHING": 0,
                             "LG": 1,
                             "ET": 2,
                             "LG+ET": 3}
        self.game_actions_idx = {v: k for k, v in self.game_actions.items()}
        self.action_space = {i: Discrete(len(self.game_actions)) for i in self.agents}
        self.observation_space = torch.tensor([0.5, 0.5, 0.5, 10.0 / 20]).repeat(self.num_agents, 1)  # TODO sort this out basos

        """
        This values define the planetary boundaries of the AYS model
        """
        self.A_PB = torch.tensor([self._compactification(ays.boundary_parameters["A_PB"], self.X_MID[0])]).repeat(
            self.num_agents, 1)  # Planetary boundary: 0.5897
        self.Y_SF = torch.tensor([self._compactification(ays.boundary_parameters["W_SF"], self.X_MID[1])]).repeat(
            self.num_agents, 1)  # Social foundations as boundary: 0.3636
        self.E_LIMIT = torch.tensor([1.0]).repeat(self.num_agents, 1)
        self.PB = torch.cat((self.E_LIMIT,
                             self.Y_SF,
                             self.A_PB), dim=1)  # EYA
        self.PB_2 = torch.cat((self.A_PB,
                               self.Y_SF,
                               torch.tensor([0.0]).repeat(self.num_agents, 1)), dim=1)  # AYS
        self.PB_3 = torch.cat((torch.tensor([0.0]).repeat(self.num_agents, 1),
                               self.Y_SF,
                               torch.tensor([0.0]).repeat(self.num_agents, 1),
                               torch.tensor([0.0]).repeat(self.num_agents, 1)), dim=1)  # AYSE
        self.PB_4 = torch.cat((self.A_PB,
                               torch.tensor([0.0]).repeat(self.num_agents, 1),
                               torch.tensor([0.0]).repeat(self.num_agents, 1)), dim=1)  # AYS

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey):
        limit = 0.05
        # heterogeneous
        state = jrandom.uniform(key, (self.num_agents, 4), minval=0.5 - limit, maxval=0.5 + limit)
        state = state.at[:, 2].set(0.5)
        state = state.at[:, 0].set(state[0, 0])

        # homogeneous
        # state = jrandom.uniform(key, (1, 3), minval=0.5 - limit, maxval=0.5 + limit)
        # state = state.at[0, 2].set(0.5)
        # state = jnp.tile(state, self.num_agents).reshape((self.num_agents, 3))

        state = state.at[:, 3].set(0)  # TODO setting emissions to zero at first step, need to add the calc here in future

        assert jnp.all(state[:, 0] == state[0, 0]), "A - First column values are not equal."

        env_state = EnvState(ayse=state,
                             terminal=jnp.array(False),
                             step=0)

        return self._get_obs(env_state), env_state

    @partial(jax.jit, static_argnums=(0,))
    def _get_obs(self, state: EnvState) -> Dict:
        # should do partial and full obs options maybe?
        return state  # TODO reimplement this a lot better

    @partial(jax.jit, static_argnums=(0,))
    def step(self,
             key: chex.PRNGKey,
             state: EnvState,
             actions: Dict[str, chex.Array],
             ) -> Tuple[Dict[str, chex.Array], EnvState, Dict[str, float], Dict[str, bool], Dict]:
        """Performs step transitions in the environment."""

        key, key_reset = jax.random.split(key)
        obs_st, states_st, rewards, dones, infos = self.step_env(key, state, actions)

        obs_re, states_re = self.reset(key_reset)

        # Auto-reset environment based on termination
        states = jax.tree_map(lambda x, y: jax.lax.select(dones["__all__"], x, y), states_re, states_st)
        obs = jax.tree_map(lambda x, y: jax.lax.select(dones["__all__"], x, y), obs_re, obs_st)

        return obs, states, rewards, dones, infos

    @partial(jax.jit, static_argnums=(0,))
    def step_env(self, key: chex.PRNGKey, state: EnvState, actions: dict):
        actions = jnp.array([actions[i] for i in self.agents])
        ayse = state.ayse

        step = state.step + 1  # TODO should it be 1 or dt

        action_matrix = self._get_parameters(actions)

        traj_one_step = odeint(self._AYS_rescaled_rhs_marl, ayse, jnp.array([state.step, step], dtype=jnp.float32),
                               action_matrix, mxstep=50000)
        # TODO results match if it is using x64 bit precision but basically close with x32, worth mentioning in paper

        new_state = traj_one_step[1]
        new_obs = self._get_obs(new_state)
        assert jnp.all(new_state[:, 0] == new_state[0, 0]), "A - First column values are not equal."  # TODO works if don't call early stop stuff if you remember

        # check if terminal, think we do total terminal or not
        rewards = self._get_rewards()

        # do the agent dones as well idk if it be useful for later on
        dones = {agent: self._inside_planetary_boundaries(new_state, agent_index) for agent_index, agent in enumerate(self.agents)}  # TODO there may be a way to avoid indexs and instaed use the agent names here

        # reward function innit
        rewards = "asdf"

        # for agent in range(self.num_agents):
        #     if self._arrived_at_final_state(agent):
        #         self.final_state[agent] = True
        #     if self.reward_type[agent] == "PB":  # or self.reward_type[agent] == "PB_new_new_new_new":
        #         if not self._inside_planetary_boundaries(agent):
        #             self.final_state[agent] = True
        #     else:
        #         if not self._inside_A_pb(agent):
        #             self.final_state[agent] = True
        #
        # # if not self.trade_actions:  # if using trade actions then this does not apply as the reward functions may not use same definition of reaching a final state  # this is the One Stop strategy
        # #     if torch.any(self.final_state):
        # #         for agent in range(self.num_agents):
        # #             if self.final_state[agent]:
        # #                 if self.green_fixed_point(agent):
        # #                     pass
        # #                 else:
        # #                     self.final_state = torch.tensor([True]).repeat(self.num_agents, 1)
        #
        # if torch.all(self.final_state):
        #     for agent in range(self.num_agents):
        #         if self.trade_actions:
        #             if self.reward_type[agent] == "PB" or self.reward_type[agent] == "IPB":
        #                 e = self.state[agent, 0]
        #                 y = self.state[agent, 1]
        #                 a = self.state[agent, 2]
        #                 if torch.abs(e - self.green_fp[agent, 0]) < self.final_radius[agent] \
        #                         and torch.abs(y - self.green_fp[agent, 1]) < self.final_radius[agent] \
        #                         and torch.abs(a - self.green_fp[agent, 2]) < self.final_radius[agent]:
        #                     self.reward[agent] += 10
        #
        #         self.reward[agent] += self.calculate_expected_final_reward(agent)
        # else:
        #     self.final_state = torch.tensor([False]).repeat(self.num_agents, 1)

        # set all done if the state is terminal
        dones["__all__"] = self._inside_planetary_boundaries_all(new_state)

        # add infos
        info = "returns maybe idk"  # TODO think need to add some element of win ratio here so can figure out if agent wins or not

        env_state = EnvState(ayse=new_state,
                             terminal=jnp.array(False),  # TODO adjust this thing
                             step=step)

        return new_obs, env_state, rewards, dones, info

    @partial(jax.jit, static_argnums=(0,))
    def _get_parameters(self, actions):

        """
        This function is needed to return the parameter set for the chosen management option.
        Here the action numbers are really transformed to parameter lists, according to the chosen
        management option.
        Parameters:
            -action_number: Number of the action in the actionset.
             Can be transformed into: 'default', 'degrowth' ,'energy-transformation' or both DG and ET at the same time
        """
        tau_A = 50  # carbon decay - single val
        tau_S = 50  # renewable knowledge stock decay - multi val
        beta = 0.03  # economic output growth - multi val
        beta_LG = 0.015  # halved economic output growth - multi val
        eps = 147  # energy efficiency param - single val
        A_offset = 600
        theta = beta / (950 - A_offset)  # beta / ( 950 - A_offset(=350) )  # theta = 8.57e-5
        rho = 2.  # renewable knowledge learning rate - multi val
        sigma = 4e12  # break even knowledge - multi val
        sigma_ET = sigma * 0.5 ** (1 / rho)  # can't remember the change, but it's somewhere - multi val
        phi = 4.7e10

        action_0 = jnp.array((beta, eps, phi, rho, sigma, tau_A, tau_S, theta))
        action_1 = jnp.array((beta_LG, eps, phi, rho, sigma, tau_A, tau_S, theta))
        action_2 = jnp.array((beta, eps, phi, rho, sigma_ET, tau_A, tau_S, theta))
        action_3 = jnp.array((beta_LG, eps, phi, rho, sigma_ET, tau_A, tau_S, theta))

        poss_action_matrix = jnp.array([action_0, action_1, action_2, action_3])

        return poss_action_matrix[actions, :]

    @partial(jax.jit, static_argnums=(0,))
    def _ays_rescaled_rhs_marl(self, ayse, t, args):
        """
        beta    = 0.03/0.015 = args[0]
        epsilon = 147        = args[1]
        phi     = 4.7e10     = args[2]
        rho     = 2.0        = args[3]
        sigma   = 4e12/sigma * 0.5 ** (1 / rho) = args[4]
        tau_A   = 50         = args[5]
        tau_S   = 50         = args[6]
        theta   = beta / (950 - A_offset) = args[7]
        # trade = args[8]
        """
        A_mid = 250  # TODO should these change with the different starting points though? idk yikes !!!I think so!!!
        Y_mid = 7e13
        S_mid = 5e11
        E_mid = 10.01882267  # TODO is this correct idk?
        # TODO how will this work with multi-agent though, could this be an input, but then again how would it work marl?

        ays_inv_matrix = 1 - ayse
        # inv_s_rho = ays_inv_matrix.copy()
        inv_s_rho = ays_inv_matrix.at[:, 2].power(args[:, 3])
        # A_matrix = (A_mid * ays_matrix[:, 0] / ays_inv_matrix[:, 0]).view(2, 1)

        # Normalise
        A_matrix = A_mid * (ayse[:, 0] / ays_inv_matrix[:, 0])
        Y_matrix = Y_mid * (ayse[:, 1] / ays_inv_matrix[:, 1])
        G_matrix = inv_s_rho[:, 2] / (inv_s_rho[:, 2] + (S_mid * ayse[:, 2] / args[:, 4]) ** args[:, 3])
        E_matrix = G_matrix / (args[:, 2] * args[:, 1]) * Y_matrix
        E_tot = jnp.sum(E_matrix) / E_matrix.shape[0]  # TODO added divide by agents, need to check is correct response

        adot = (E_tot - (A_matrix / args[:, 5])) * ays_inv_matrix[:, 0] * ays_inv_matrix[:,
                                                                          0] / A_mid  # TODO check this maths
        # adot = G_matrix / (args[:, 2] * args[:, 1] * A_mid) * ays_inv_matrix[:, 0] * ays_inv_matrix[:, 0] * Y_matrix - ayse[:, 0] * ays_inv_matrix[:, 0] / args[:, 5]
        ydot = ayse[:, 1] * ays_inv_matrix[:, 1] * (args[:, 0] - args[:, 7] * A_matrix)
        sdot = (1 - G_matrix) * ays_inv_matrix[:, 2] * ays_inv_matrix[:, 2] * Y_matrix / (args[:, 1] * S_mid) - ayse[:,
                                                                                                                2] * ays_inv_matrix[
                                                                                                                     :,
                                                                                                                     2] / args[
                                                                                                                          :,
                                                                                                                          6]

        E_output = E_matrix / (E_matrix + E_mid)

        return jnp.concatenate(
            (adot[:, jnp.newaxis], ydot[:, jnp.newaxis], sdot[:, jnp.newaxis], E_output[:, jnp.newaxis]), axis=1)

    @partial(jax.jit, static_argnums=(0,))
    def _get_rewards(self, action=None):
        def reward_distance_PB(agent):
            if self._inside_planetary_boundaries(agent):
                self.reward[agent] = torch.norm(self.reward_space[agent, :3] - self.PB_2[agent])
            else:
                self.reward[agent] = 0.0

        def incentivised_reward_distance_PB(agent):
            if self._inside_A_pb(agent):
                new_reward = torch.norm(self.reward_space[agent, :3] - self.PB_4[agent])
            #     #  new_reward = torch.sqrt(4 * (torch.abs(self.reward_space[agent, 0] - self.PB_4[agent, 0])) ** 2 + (torch.abs(self.reward_space[agent, 1] - self.PB_4[agent, 1])) ** 2 + (torch.abs(self.reward_space[agent, 2] - self.PB_4[agent, 2])) ** 2)  # weighted by 4 to the a goal
            else:
                new_reward = 0.0

            if self.old_reward[agent] > new_reward:
                self.reward[agent] = -new_reward
            elif self.old_reward[agent] < new_reward:
                self.reward[agent] = new_reward
            else:
                self.reward[agent] = 0.0

            self.old_reward[agent] = new_reward

        def reward_distance_Y(agent):
            self.reward[agent] = torch.abs(self.reward_space[agent, 1] - self.PB_3[agent, 1])  # max y

        def reward_distance_E(agent):
            self.reward[agent] = torch.abs(self.reward_space[agent, 3] - self.PB_3[agent, 3])  # max e

        def reward_distance_A(agent):
            self.reward[agent] = torch.abs(self.reward_space[agent, 0] - self.PB_3[agent, 0]) / 4  # max a

        for agent in range(self.num_agents):
            if self.reward_type[agent] == 'PB':
                reward_distance_PB(agent)
            elif self.reward_type[agent] == 'IPB':
                incentivised_reward_distance_PB(agent)
            elif self.reward_type[agent] == 'max_Y':
                reward_distance_Y(agent)
            elif self.reward_type[agent] == 'max_E':
                reward_distance_E(agent)
            elif self.reward_type[agent] == 'max_A':
                reward_distance_A(agent)
            elif self.reward_type[agent] is None:
                print("ERROR! You have to choose a reward function!\n",
                      "Available Reward functions for this environment are: PB, rel_share, survive, desirable_region!")
                exit(1)
            else:
                print("ERROR! The reward function you chose is not available! " + self.reward_type[agent])
                self.print_debug_info()
                sys.exit(1)

        return "rewards"  # TODO implement rewards all in one somehow

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

        return discounted_future_reward

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

    def _inside_A_pb(self, agent):
        e = self.state[agent, 0]
        y = self.state[agent, 1]
        a = self.state[agent, 2]
        is_inside = True

        if a > self.A_PB[agent]:
            is_inside = False
            # print("Outside PB!")
        return is_inside

    @partial(jax.jit, static_argnums=(0,))
    def _inside_planetary_boundaries(self, ayse, agent_index):  # TODO can we subsume all into the individual somehow
        e = ayse[agent_index, 3]
        y = ayse[agent_index, 1]
        a = ayse[agent_index, 0]
        is_inside = True

        if a > self.A_PB[agent_index] or y < self.Y_SF[agent_index] or e > self.E_LIMIT[agent_index]:
            is_inside = False
        return is_inside

    @partial(jax.jit, static_argnums=(0,))
    def _inside_planetary_boundaries_all(self, ayse):
        e = ayse[:, 3]
        y = ayse[:, 1]
        a = ayse[:, 0]
        is_inside = True

        if jnp.all(a > self.A_PB) or jnp.all(y < self.Y_SF) or jnp.all(e > self.E_LIMIT):
            is_inside = False
        return is_inside

    def _arrived_at_final_state(self, agent):
        e = self.state[agent, 0]
        y = self.state[agent, 1]
        a = self.state[agent, 2]

        if torch.abs(e - self.green_fp[agent, 0]) < self.final_radius[agent] \
                and torch.abs(y - self.green_fp[agent, 1]) < self.final_radius[agent] \
                and torch.abs(a - self.green_fp[agent, 2]) < self.final_radius[agent]:
            return True
        elif torch.abs(e - self.brown_fp[agent, 0]) < self.final_radius[agent] \
                and torch.abs(y - self.brown_fp[agent, 1]) < self.final_radius[agent] \
                and torch.abs(a - self.brown_fp[agent, 2]) < self.final_radius[agent]:
            return True
        else:
            return False

    def green_fixed_point(self, agent):
        e = self.state[agent, 0]
        y = self.state[agent, 1]
        a = self.state[agent, 2]

        if torch.abs(e - self.green_fp[agent, 0]) < self.final_radius[agent] \
                and torch.abs(y - self.green_fp[agent, 1]) < self.final_radius[agent] \
                and torch.abs(a - self.green_fp[agent, 2]) < self.final_radius[agent]:
            return True
        else:
            return False

    def _good_final_state(self, agent):
        e = self.state[agent, 0]
        y = self.state[agent, 1]
        a = self.state[agent, 2]
        if np.abs(e - self.green_fp[agent, 0]) < self.final_radius[agent] \
                and np.abs(y - self.green_fp[agent, 1]) < self.final_radius[agent] \
                and np.abs(a - self.green_fp[agent, 2]) < self.final_radius[agent]:
            return True
        else:
            return False

    def which_final_state(self, agent):
        e = self.state[agent, 0]
        y = self.state[agent, 1]
        a = self.state[agent, 2]
        if np.abs(a - self.green_fp[agent, 0]) < self.final_radius[agent] and np.abs(y - self.green_fp[agent, 1]) < \
                self.final_radius[agent] and np.abs(a - self.green_fp[agent, 2]) < self.final_radius[agent]:
            # print("ARRIVED AT GREEN FINAL STATE WITHOUT VIOLATING PB!")
            return Basins.GREEN_FP
        elif np.abs(a - self.brown_fp[agent, 0]) < self.final_radius[agent] and np.abs(y - self.brown_fp[agent, 1]) < \
                self.final_radius[agent] and np.abs(a - self.brown_fp[agent, 2]) < self.final_radius[agent]:
            return Basins.BLACK_FP
        else:
            # return Basins.OUT_PB
            return self._which_PB(agent)

    def _which_PB(self, agent):
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

    def get_plot_state_list(self):
        return self.state, self.reward_space


def example():
    num_agents = 5
    key = jax.random.PRNGKey(0)

    env = AYS_Environment()

    obs, state = env.reset(key)
    # env.render(state)

    for _ in range(20):
        key, key_reset, key_act, key_step = jax.random.split(key, 4)

        # env.render(state)
        # print("obs:", obs)

        # Sample random actions.
        key_act = jax.random.split(key_act, env.num_agents)
        actions = {agent: env.action_space[agent].sample(key_act[i]) for i, agent in enumerate(env.agents)}

        # print("action:", env.game_actions_idx[actions[env.agents[state.agent_in_room]].item()])

        # Perform the step transition.
        obs, state, reward, done, infos = env.step(key_step, state, actions)

        print("reward:", reward["agent_0"])

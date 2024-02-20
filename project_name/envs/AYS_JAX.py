"""
This is the implementation of the AYS Environment in the form
that it can used within the Agent-Environment interface
in combination with the DRL-agent.

@author: Felix Strnad, Theodore Wolf

"""

import sys
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
from jaxmarl.environments.spaces import Discrete, MultiDiscrete, Box
from jax.experimental.ode import odeint
import heapq as hq
import operator as op
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import mpl_toolkits.mplot3d as plt3d


@struct.dataclass
class EnvState:  # TODO fill this env state up as we need it basos
    ayse: chex.Array
    prev_actions: chex.Array
    dones: chex.Array
    terminal: bool
    done_causation: chex.Array
    step: int


@struct.dataclass
class InfoState:
    env_state: EnvState
    episode_returns: float
    episode_lengths: int
    won_episode: int
    returned_episode_returns: float
    returned_episode_lengths: int
    returned_won_episode: int


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
        self.agent_ids = {agent: i for i, agent in enumerate(self.agents)}  # TODO use this more

        self.final_state = jnp.tile(jnp.array([False]), self.num_agents)
        self.reward_type = reward_type  # TODO is there a better way to do this with agent names
        print(f"Reward type: {self.reward_type}")

        self.timeStart = 0
        self.intSteps = 10  # integration Steps
        self.t = self.t0 = t0
        self.dt = dt
        self.sim_time_step = jnp.linspace(self.timeStart, dt, self.intSteps)

        self.green_fp = jnp.array([0.0, 1.0, 0.0, 0.0])  # ayse
        self.black_fp = jnp.array([0.6, 0.4, 0.0, 1.0])  # ayse idk what e should be really
        self.final_radius = jnp.array([0.05])
        self.color_list = ays_plot.color_list

        self.game_actions = {"NOTHING": 0,
                             "LG": 1,
                             "ET": 2,
                             "LG+ET": 3}
        self.game_actions_idx = {v: k for k, v in self.game_actions.items()}
        self.action_space = {i: Discrete(len(self.game_actions)) for i in self.agents}
        self.observation_space = {i: Box(low=0.0, high=1.0, shape=(5,)) for i in
                                  self.agents}  # TODO are you meant to add actions to obervation space I forget

        """
        This values define the planetary boundaries of the AYS model
        """
        self.start_point = [240, 7e13, 501.5198]  # TODO should this change, also why is it 501.5198 i cant remember
        self.A_offset = 600
        self.A_boundary_param = 945 - self.A_offset
        self.A_PB = jnp.array([self._compactification(ays.boundary_parameters["A_PB"],
                                                      self.start_point[0])])  # Planetary boundary: 0.5897
        self.Y_PB = jnp.array(([self._compactification(ays.boundary_parameters["W_SF"],
                                                       self.start_point[1])]))  # Social foundations as boundary: 0.3636
        self.S_LIMIT = jnp.array([0.0])  # i.e. min value we want
        self.E_LIMIT = jnp.array([1.0])  # i.e. max value we want

        self.PB = jnp.concatenate((self.A_PB, self.Y_PB, self.E_LIMIT))  # AYE
        self.PB_2 = jnp.concatenate((self.A_PB, self.Y_PB, self.S_LIMIT))  # AYS
        self.PB_3 = jnp.concatenate(
            (jnp.array([0.0]), jnp.array([0.0]), self.S_LIMIT, jnp.array([0.0])))  # AYSE negative behaviour
        self.PB_4 = jnp.concatenate((self.A_PB, jnp.array([0.0]), self.S_LIMIT))  # AYS

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

        state = state.at[:, 3].set(0)  # TODO setting emissions to zero at first step, need to add calc here in future

        actions = jnp.array([0 for _ in self.agents])
        traj_one_step = odeint(self._ays_rescaled_rhs_marl, state,
                               jnp.array([0.0, 1.0], dtype=jnp.float32),
                               self._get_parameters(actions), mxstep=50000)
        state = state.at[:, 3].set(traj_one_step[1, :, 3])  # way to prevent first e being 0 by running one calc

        # assert jnp.allclose(state.at[:, 0].get(), state.at[0, 0].get()), "A - First column values are not equal."  # TODO reimplement this to work

        env_state = EnvState(ayse=state,
                             prev_actions=jnp.zeros((self.num_agents,), dtype=jnp.int32),
                             dones={agent: jnp.array(False) for agent in ["__all__"] + self.agents},
                             terminal=jnp.array(False),
                             done_causation={agent: 0 for agent in self.agents},
                             step=jnp.array(0))
        wrapper_state = InfoState(env_state,
                                  jnp.zeros((self.num_agents,)),
                                  jnp.zeros(1),
                                  0.0,
                                  jnp.zeros((self.num_agents,)),
                                  jnp.zeros(1),
                                  jnp.zeros(1),
                                  )

        graph_state = jnp.zeros((self.max_steps, self.num_agents, 4))
        graph_state = graph_state.at[0, :].set(state)

        return self._get_obs(env_state), wrapper_state, graph_state

    @partial(jax.jit, static_argnums=(0,))
    def _get_obs(self, env_state: EnvState) -> Dict:
        # should do partial and full obs options maybe?
        full_obs = jnp.concatenate((env_state.ayse, env_state.prev_actions.reshape(-1, 1)), axis=1)
        return {agent: full_obs[self.agent_ids[agent]] for agent in self.agents}

    @partial(jax.jit, static_argnums=(0,))
    def step(self,
             key: chex.PRNGKey,
             state: InfoState,
             actions: Dict[str, chex.Array],
             graph_state: chex.Array
             ) -> Tuple[Dict[str, chex.Array], InfoState, Dict[str, float], Dict[str, bool], Dict]:
        """Performs step transitions in the environment."""

        key, key_reset = jax.random.split(key)
        obs_st, states_st, rewards, dones, infos, graph_states_st = self.step_env(key, state, actions, graph_state)

        obs_re, states_re, graph_states_re = self.reset(key_reset)

        # Auto-reset environment based on termination
        states = jax.tree_map(lambda x, y: jax.lax.select(dones["__all__"], x, y), states_re, states_st)
        obs = jax.tree_map(lambda x, y: jax.lax.select(dones["__all__"], x, y), obs_re, obs_st)
        graph_states = jax.tree_map(lambda x, y: jax.lax.select(dones["__all__"], x, y), graph_states_re,
                                    graph_states_st)

        # add info stuff idk if right spot
        def _batchify_floats(x: dict):
            return jnp.stack([x[a] for a in self.agents])

        ep_done = dones["__all__"]
        batch_reward = _batchify_floats(rewards)
        new_episode_return = state.episode_returns + _batchify_floats(rewards)
        new_episode_length = state.episode_lengths + 1
        new_won_episode = jnp.any(
            jnp.array([infos["agent_done_causation"][agent] for agent in self.agents]) == 1).astype(
            dtype=jnp.float32)  # TODO check this works - also make sure doesn't randomly say 1 and uses another win
        # TODO need to think about if both agents end etc if the above would work
        wrapper_state = InfoState(env_state=states.env_state,
                                  won_episode=new_won_episode * (1 - ep_done),
                                  episode_returns=new_episode_return * (1 - ep_done),
                                  episode_lengths=new_episode_length * (1 - ep_done),
                                  returned_episode_returns=state.returned_episode_returns * (
                                          1 - ep_done) + new_episode_return * ep_done,
                                  returned_episode_lengths=state.returned_episode_lengths * (
                                          1 - ep_done) + new_episode_length * ep_done,
                                  returned_won_episode=state.returned_won_episode * (
                                          1 - ep_done) + new_won_episode * ep_done,
                                  )
        infos["returned_episode_returns"] = state.returned_episode_returns
        infos["returned_episode_lengths"] = state.returned_episode_lengths
        infos["returned_won_episode"] = state.returned_won_episode
        infos["returned_episode"] = jnp.full((self.num_agents,), ep_done)

        return obs, wrapper_state, rewards, dones, infos, graph_states

    @partial(jax.jit, static_argnums=(0,))
    def step_env(self, key: chex.PRNGKey, state: InfoState, actions: dict, graph_state: chex.Array) -> Tuple[
        Dict[str, chex.Array], InfoState, Dict[str, float], Dict[str, bool], Dict]:
        actions = jnp.array([actions[i] for i in self.agents])

        step = state.env_state.step + 1  # TODO should it be 1 or dt

        action_matrix = self._get_parameters(actions).squeeze(axis=1)

        input_state = state.env_state.ayse.at[:, 3].set(0.0)  # resets emissions to 0 otherwise it effects the sim
        traj_one_step = odeint(self._ays_rescaled_rhs_marl, input_state,
                               jnp.array([state.env_state.step, step], dtype=jnp.float32),
                               action_matrix, mxstep=50000)
        # TODO results match if it is using x64 bit precision but basically close with x32, worth mentioning in paper

        new_state = traj_one_step[1]
        # assert jnp.all(new_state[:, 0] == new_state[0, 0]), "A - First column values are not equal."  # TODO make the above work
        # TODO above works if don't call early stop stuff if you remember

        graph_state = graph_state.at[step, :].set(new_state)  # TODO check the step thing works and is correctly defined
        # print(step)
        # print(graph_state)

        env_state = state.env_state

        env_state = env_state.replace(ayse=new_state,
                                      prev_actions=actions.squeeze(),
                                      step=step,
                                      terminal=self._terminal_state(new_state, step),
                                      # TODO check terminal works basically for both cases and whether it is being called on the right state
                                      )

        # convert state to obs
        obs = self._get_obs(env_state)

        # do the agent dones as well idk if it be useful for later on
        dones = {agent: jnp.logical_or(~self._inside_planetary_boundaries(new_state, self.agent_ids[agent]),
                                       self._arrived_at_final_state(new_state, self.agent_ids[agent])) for agent in
                 self.agents}  # TODO check the tilde actually works

        # reward function innit
        rewards = self._get_rewards(env_state.ayse)

        # for agent in self.agents:  # TODO is this okay idk? do we want expected final reward
        #     if dones[agent]:
        #         rewards = rewards.at(self.agent_ids[agent]).set(self._calculate_expected_final_reward(rewards, self.agent_ids[agent], step))  # TODO is this step right here or should it be last step? also can this be improved if used

        dones["__all__"] = env_state.terminal
        env_state = env_state.replace(dones=dones)
        env_state = self._done_causation(env_state)

        # add infos
        info = {"agent_done_causation": {agent: env_state.done_causation[agent] for agent in self.agents}}
        state = state.replace(env_state=env_state)

        return (jax.lax.stop_gradient(obs),
                jax.lax.stop_gradient(state),
                rewards,
                dones,
                info,
                jax.lax.stop_gradient(graph_state))

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

        E_output = E_matrix / (E_matrix + E_mid)  # TODO sort this out as hardcoding is bAD

        return jnp.concatenate(
            (adot[:, jnp.newaxis], ydot[:, jnp.newaxis], sdot[:, jnp.newaxis], E_output[:, jnp.newaxis]), axis=1)

    @partial(jax.jit, static_argnums=(0,))
    def _get_rewards(self, ayse):
        def reward_distance_PB(agent):
            # print(self._inside_planetary_boundaries(ayse, agent))
            # print(type(self._inside_planetary_boundaries(ayse, agent)))
            # sys.exit()
            return jnp.where(self._inside_planetary_boundaries(ayse, agent),
                             jnp.linalg.norm(ayse[agent, :3] - self.PB_2),
                             0.0)

        # def incentivised_reward_distance_PB(agent):  # TODO need to reimplement somehow with old rewards
        #     if self._inside_A_pb(agent):
        #         new_reward = jnp.linalg.norm(ayse[agent, :3] - self.PB_4[agent])
        #     #     #  new_reward = torch.sqrt(4 * (torch.abs(self.reward_space[agent, 0] - self.PB_4[agent, 0])) ** 2 + (torch.abs(self.reward_space[agent, 1] - self.PB_4[agent, 1])) ** 2 + (torch.abs(self.reward_space[agent, 2] - self.PB_4[agent, 2])) ** 2)  # weighted by 4 to the a goal
        #     else:
        #         new_reward = 0.0
        #
        #     if self.old_reward[agent] > new_reward:
        #         self.reward[agent] = -new_reward
        #     elif self.old_reward[agent] < new_reward:
        #         self.reward[agent] = new_reward
        #     else:
        #         self.reward[agent] = 0.0
        #
        #     self.old_reward[agent] = new_reward

        def reward_distance_Y(agent):
            return jnp.abs(ayse[agent, 1] - self.PB_3[1])  # max y

        def reward_distance_E(agent):
            return jnp.abs(ayse[agent, 3] - self.PB_3[3])  # max e

        def reward_distance_A(agent):
            return jnp.abs(ayse[agent, 0] - self.PB_3[0])  # max a

        rewards = jnp.zeros(self.num_agents)
        for agent in self.agents:
            agent_index = self.agent_ids[agent]
            if self.reward_type[agent_index] == 'PB':
                agent_reward = reward_distance_PB(agent_index)
            # elif self.reward_type[agent] == 'IPB':
            #     agent_reward = incentivised_reward_distance_PB(agent)
            elif self.reward_type[agent_index] == 'max_Y':
                agent_reward = reward_distance_Y(agent_index)
            elif self.reward_type[agent_index] == 'max_E':
                agent_reward = reward_distance_E(agent_index)
            elif self.reward_type[agent_index] == 'max_A':
                agent_reward = reward_distance_A(agent_index)
            else:
                print("ERROR! The reward function you chose is not available! " + self.reward_type[agent_index])
                sys.exit()
            rewards = rewards.at[agent_index].set(agent_reward.squeeze())  # TODO added a squeeze idk if right

        return {agent: rewards[self.agent_ids[agent]] for agent in self.agents}

    @partial(jax.jit, static_argnums=(0,))
    def _calculate_expected_final_reward(self, rewards, agent, step):
        """
        Get the reward in the last state, expecting from now on always default.
        This is important since we break up simulation at final state, but we do not want the agent to
        find trajectories that are close (!) to final state and stay there, since this would
        result in a higher total reward.
        """
        remaining_steps = self.max_steps - step
        discounted_future_reward = 0.
        for i in range(remaining_steps):
            discounted_future_reward += self.gamma ** i * rewards[self.agent_ids[agent]]

        return discounted_future_reward

    @partial(jax.jit, static_argnums=(0,))
    def _compactification(self, x, x_mid):  # TODO check this works
        return jnp.select([x == 0, x == np.infty],
                          [0.0, 1.0], x / (x + x_mid))

    @partial(jax.jit, static_argnums=(0,))
    def _inv_compactification(self, y, x_mid):  # TODO check this works
        return jnp.select([y == 0, jnp.allclose(y, 1)],
                          [0.0, np.infty], x_mid * y / (1 - y))

    @partial(jax.jit, static_argnums=(0,))
    def _inside_planetary_boundaries(self, ayse, agent_index):  # TODO can we subsume all into the individual somehow
        result = jax.lax.select(jnp.logical_and(ayse.at[agent_index, 0].get() < self.A_PB,
                                                ayse.at[agent_index, 1].get() > self.Y_PB,
                                                ).squeeze(), True, False)

        return result

    @partial(jax.jit, static_argnums=(0,))
    def _terminal_state(self, ayse, step):  # TODO confirm this works
        # True should flow through no matter what
        result = jax.lax.select(jnp.logical_or(step >= self.max_steps,
                                                jnp.logical_or(jnp.all(ayse.at[:, 0].get() >= self.A_PB),
                                                                jnp.all(ayse.at[:, 1].get() <= self.Y_PB),
                                                                ).squeeze()), True, False)

        # TODO add in belowAdding also if at a fixed point green or black
        # for agent in self.agents:
        #     result = self._arrived_at_final_state(ayse, self.agent_ids[agent], result)  # TODO means if one agent finish then all finish
        # TODO can turn the above into a scan?

        return result

    @partial(jax.jit, static_argnums=(0,))
    def _arrived_at_final_state(self, ayse, agent_index):  # TODO confirm this works
        result = jax.lax.select(jnp.logical_or(jnp.all(jnp.abs(ayse - self.green_fp)[agent_index, :] < self.final_radius),
                                               jnp.all(jnp.abs(ayse - self.black_fp)[agent_index, :].all() < self.final_radius)).squeeze(),
                               True, False)

        return result

        # if jnp.all(jnp.abs(ayse - self.green_fp)[agent_index, :] < self.final_radius):
        #     return True
        # elif jnp.all(jnp.abs(ayse - self.black_fp)[agent_index, :] < self.final_radius):
        #     return True
        # else:
        #     return False

    @partial(jax.jit, static_argnums=(0,))
    def _green_fixed_point(self, ayse, agent_index):  # TODO can we add this to the above to save code space
        return jax.lax.select(jnp.all(jnp.abs(ayse - self.green_fp)[agent_index, :] < self.final_radius),
                              True,
                              False)  # TODO confirm this works

    @partial(jax.jit, static_argnums=(0,))
    def _black_fixed_point(self, ayse, agent_index):  # TODO can we add this to the above to save code space
        return jax.lax.select(jnp.all(jnp.abs(ayse - self.black_fp)[agent_index, :] < self.final_radius),
                              True,
                              False)  # TODO confirm this works

    @partial(jax.jit, static_argnums=(0,))
    def _which_final_state(self, ayse, agent_index):  # TODO same here can we add the green fixed point tINGG
        return jnp.select([self._green_fixed_point(ayse, agent_index), self._black_fixed_point(ayse, agent_index)],
                          [1, 2], self._which_PB(ayse, agent_index))

        # if self._green_fixed_point(ayse, agent_index):
        #     return 1
        # elif self._black_fixed_point(ayse, agent_index):
        #     return 2
        # else:
        #     return self._which_PB(ayse, agent_index)

    @partial(jax.jit, static_argnums=(0,))
    def _which_PB(self, ayse, agent_index):
        return jnp.select([ayse[agent_index, 0] >= self.A_PB,
                           ayse[agent_index, 1] <= self.Y_PB,
                           ayse[agent_index, 2] <= self.S_LIMIT],
                          [3, 4, 5], 7)

        # if ayse[agent_index, 0] >= self.A_PB:
        #     return 3
        # elif ayse[agent_index, 1] <= self.Y_PB:
        #     return 4
        # elif ayse[agent_index, 2] <= self.S_LIMIT:  # TODO check this ting
        #     return 5
        # # elif ayse[agent_index, 3] >= self.E_LIMIT:  # TODO do we need an e limit idk?
        # #     return 6
        # else:
        #     return 7

    @partial(jax.jit, static_argnums=(0,))
    def _done_causation(self,
                        state: EnvState):  # TODO check this works - it says black fixed point but this not always true, cus of the or function
        """
        Parameters
        ----------
        state

        Returns
        -------
        0 = None
        1 = Green Fixed Point
        2 = Black Fixed Point
        3 = A_PB
        4 = Y_PB
        5 = S_LIMIT
        6 = E_LIMIT
        7 = Out_Of_Time

        """
        for agent in self.agents:
            state.done_causation[agent] = jnp.where(state.dones[agent],
                                                    # TODO there probs a better way to write this than another for loop
                                                    self._which_final_state(state.ayse, self.agent_ids[agent]),
                                                    0).squeeze()

        return state

    @partial(jax.jit, static_argnums=(0,))
    def render(self, graph_states: chex.Array):
        fig, ax3d = create_figure_ays(top_down=False)  # TODO sort the self stuff first
        colors = plt.cm.brg(np.linspace(0, 1, self.num_agents))
        # print(graph_states)
        graph_states = graph_states[~(jnp.all(graph_states == 0.0, axis=(1, 2)))]  # TODO add something to cut out the bad bits from graphing
        for agent in self.agents:
            ax3d.plot3D(xs=graph_states[:, self.agent_ids[agent], 3],
                        ys=graph_states[:, self.agent_ids[agent], 1],
                        zs=graph_states[:, self.agent_ids[agent], 0],
                        color=colors[self.agent_ids[agent]],
                        alpha=0.8, lw=3, label=f"{agent}")
        # ax3d.set_title([f"{agent} {self.reward_type[self.agent_ids[agent]]} reward : {reward_values[agent][num]:.2f}" for
        #      agent in self.agents)])
        # image = Image.frombytes("RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        # return image

        plt.savefig(f"project_name/images/{graph_states.shape[0]}.png")
        plt.close()

        return

    @partial(jax.jit, static_argnums=(0,))
    def _create_figure_ays(self, top_down, label=None, colors=None, ax=None, ticks=True, plot_boundary=True,
                           reset=False, ax3d=None, fig3d=None):
        color_list = ['#e41a1c', '#ff7f00', '#4daf4a', '#377eb8', '#984ea3']
        if not reset:
            if ax is None:
                fig3d = plt.figure(figsize=(10, 8))
                # ax3d = plt3d.Axes3D(fig3d)
                ax3d = fig3d.add_subplot(111, projection="3d")
            else:
                ax3d = ax
                fig3d = None

        if ticks == True:
            self._make_3d_ticks_ays(ax3d)
        else:
            ax3d.set_xticks([])
            ax3d.set_yticks([])
            ax3d.set_zticks([])

        A_PB = [10, 265]
        top_view = [25, 170]

        azimuth, elevation = 140, 15
        if top_down:
            azimuth, elevation = 180, 90

        ax3d.view_init(elevation, azimuth)

        S_scale = 1e9
        Y_scale = 1e12
        # ax3d.set_xlabel("\n\nexcess atmospheric carbon\nstock A [GtC]", )
        ax3d.set_xlabel("\n\nemissions E  \n  [GtC]", )
        ax3d.set_ylabel("\n\neconomic output Y \n  [%1.0e USD/yr]" % Y_scale, )
        # ax3d.set_zlabel("\n\nrenewable knowledge\nstock S [%1.0e GJ]"%S_scale,)
        if not top_down:
            ax3d.set_zlabel("\n\ntotal excess atmospheric carbon\nstock A [GtC]", )

        # Add boundaries to plot
        if plot_boundary:
            self._add_boundary(ax3d, sunny_boundaries=["planetary-boundary", "social-foundation"], model='ays',
                               **ays.grid_parameters, **ays.boundary_parameters)

        ax3d.grid(False)

        legend_elements = []
        if label is None:
            # For Management Options
            for idx in range(len(self.game_actions)):
                legend_elements.append(Line2D([0], [0], lw=2, color=color_list[idx], label=self.game_actions[idx]))

            # ax3d.scatter(*zip([0.5,0.5,0.5]), lw=1, color=shelter_color, label='Shelter')
        else:
            for i in range(len(label)):
                ax3d.scatter(*zip([0.5, 0.5, 0.5]), lw=1, color=colors[i], label=label[i])

        # For Startpoint
        # ax3d.scatter(*zip([0.5,0.5,0.5]), lw=4, color='black')

        # For legend
        legend_elements.append(
            Line2D([0], [0], lw=2, label='current state', marker='o', color='w', markerfacecolor='red', markersize=15))
        # ax3d.legend(handles=legend_elements,prop={'size': 14}, bbox_to_anchor=(0.85,.90), fontsize=20,fancybox=True, shadow=True)

        return fig3d, ax3d

    @partial(jax.jit, static_argnums=(0,))
    def _make_3d_ticks_ays(self, ax3d, boundaries=None, transformed_formatters=False, S_scale=1e9, Y_scale=1e12,
                           num_a=12,
                           num_y=12,
                           num_s=12, ):
        if boundaries is None:
            boundaries = [None] * 3

        transf = partial(self._compactification, x_mid=self.start_point[0])
        inv_transf = partial(self._inv_compactification, x_mid=self.start_point[0])

        # A- ticks
        if boundaries[0] is None:
            start, stop = 0, 20  # np.infty
            ax3d.set_xlim(0, 1)
        else:
            start, stop = inv_transf(boundaries[0])
            ax3d.set_xlim(*boundaries[0])

        ax3d.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax3d.set_xticklabels([0, 4, 8, 12, 16, 20])  # TODO is this correct bound?

        # Y - ticks
        transf = partial(self._compactification, x_mid=self.start_point[1])
        inv_transf = partial(self._inv_compactification, x_mid=self.start_point[1])

        if boundaries[1] is None:
            start, stop = 0, np.infty
            ax3d.set_ylim(0, 1)
        else:
            start, stop = inv_transf(boundaries[1])
            ax3d.set_ylim(*boundaries[1])

        formatters, locators = self._transformed_space(transf, inv_transf, axis_use=True, scale=Y_scale, start=start,
                                                       stop=stop,
                                                       num=num_y)
        if transformed_formatters:
            new_formatters = []
            for el, loc in zip(formatters, locators):
                if el:
                    new_formatters.append("{:4.2f}".format(loc))
                else:
                    new_formatters.append(el)
            formatters = new_formatters
        ax3d.yaxis.set_major_locator(ticker.FixedLocator(locators))
        ax3d.yaxis.set_major_formatter(ticker.FixedFormatter(formatters))

        transf = partial(self._compactification, x_mid=self.start_point[2])
        inv_transf = partial(self._inv_compactification, x_mid=self.start_point[2])

        # S ticks
        if boundaries[2] is None:
            start, stop = 0, np.infty
            ax3d.set_zlim(0, 1)
        else:
            start, stop = inv_transf(boundaries[2])
            ax3d.set_zlim(*boundaries[2])

        formatters, locators = self._transformed_space(transf, inv_transf, axis_use=True, start=start, stop=stop,
                                                       num=num_s)
        if transformed_formatters:
            new_formatters = []
            for el, loc in zip(formatters, locators):
                if el:
                    new_formatters.append("{:4.2f}".format(loc))
                else:
                    new_formatters.append(el)
            formatters = new_formatters
        ax3d.zaxis.set_major_locator(ticker.FixedLocator(locators))
        ax3d.zaxis.set_major_formatter(ticker.FixedFormatter(formatters))

    @partial(jax.jit, static_argnums=(0,))
    def _transformed_space(self, transform, inv_transform, start=0, stop=np.infty, num=12, scale=1, num_minors=50,
                           endpoint=True,
                           axis_use=False, boundaries=None, minors=False):
        add_infty = False
        if stop == np.infty and endpoint:
            add_infty = True
            endpoint = False
            num -= 1

        locators_start = transform(start)
        locators_stop = transform(stop)

        major_locators = np.linspace(locators_start,
                                     locators_stop,
                                     num,
                                     endpoint=endpoint)

        major_formatters = inv_transform(major_locators)
        # major_formatters = major_formatters / scale

        major_combined = list(zip(major_locators, major_formatters))
        # print(major_combined)

        if minors:
            _minor_formatters = np.linspace(major_formatters[0], major_formatters[-1], num_minors, endpoint=False)[1:]
            minor_locators = transform(_minor_formatters)
            minor_formatters = [np.nan] * len(minor_locators)
            minor_combined = list(zip(minor_locators, minor_formatters))
        # print(minor_combined)
        else:
            minor_combined = []
        combined = list(hq.merge(minor_combined, major_combined, key=op.itemgetter(0)))

        # print(combined)

        if not boundaries is None:
            combined = [(l, f) for l, f in combined if boundaries[0] <= l <= boundaries[1]]

        ret = tuple(map(np.array, zip(*combined)))
        if ret:
            locators, formatters = ret
        else:
            locators, formatters = np.empty((0,)), np.empty((0,))
        formatters = formatters / scale

        if add_infty:
            # assume locators_stop has the transformed value for infinity already
            locators = np.concatenate((locators, [locators_stop]))
            formatters = np.concatenate((formatters, [np.infty]))

        if not axis_use:
            return formatters

        else:
            string_formatters = np.zeros_like(formatters, dtype="|U10")
            mask_nan = np.isnan(formatters)
            if add_infty:
                string_formatters[-1] = u"\u221E"  # infty sign
                mask_nan[-1] = True
            string_formatters[~mask_nan] = np.round(formatters[~mask_nan], decimals=2).astype(int).astype("|U10")
            return string_formatters, locators

    @partial(jax.jit, static_argnums=(0,))
    def _add_boundary(self, ax3d, *, sunny_boundaries, add_outer=False, plot_boundaries=None, model='ays',
                      **parameters):
        """show boundaries of desirable region"""

        if not sunny_boundaries:
            # nothing to do
            return

            # get the boundaries of the plot (and check whether it's an old one where "A" wasn't transformed yet
        if plot_boundaries is None:
            if "A_max" in parameters:
                a_min, a_max = 0, parameters["A_max"]
            elif "A_mid" in parameters:
                a_min, a_max = 0, 1
            w_min, w_max = 0, 1
            s_min, s_max = 0, 1
        else:
            a_min, a_max = plot_boundaries[0]
            w_min, w_max = plot_boundaries[1]
            s_min, s_max = plot_boundaries[2]

        if model == 'ricen':
            a_min = 0
            a_max = 20
            w_min = 0
            w_max = 750
            s_min = -10
            s_max = 15

        plot_pb = False
        plot_sf = False
        if "planetary-boundary" in sunny_boundaries:
            A_PB = parameters["A_PB"]
            if "A_max" in parameters:
                pass  # no transformation necessary
            elif "A_mid" in parameters:
                A_PB = A_PB / (A_PB + parameters["A_mid"])
            else:
                assert False, "couldn't identify how the A axis is scaled"
            if a_min < A_PB < a_max:
                plot_pb = True
        if "social-foundation" in sunny_boundaries:
            W_SF = parameters["W_SF"]
            W_SF = W_SF / (W_SF + parameters["W_mid"])
            if w_min < W_SF < w_max:
                plot_sf = True

        if model == 'ricen':
            A_PB = 7

        if plot_pb and plot_sf:
            corner_points_list = [[[a_min, W_SF, A_PB],
                                   [a_min, w_max, A_PB],
                                   [a_max, w_max, A_PB],
                                   [a_max, W_SF, A_PB],
                                   ],
                                  [[a_max, W_SF, s_min],
                                   [a_min, W_SF, s_min],
                                   [a_min, W_SF, A_PB],
                                   [a_max, W_SF, A_PB],
                                   ]]
        elif plot_pb:
            corner_points_list = [
                [[a_min, w_min, A_PB], [a_min, w_max, A_PB], [a_max, w_max, A_PB], [a_max, w_min, A_PB]]]
        elif plot_sf:
            corner_points_list = [
                [[a_min, W_SF, s_min], [a_max, W_SF, s_min], [a_max, W_SF, s_max], [a_min, W_SF, s_max]]]
        else:
            raise ValueError("something wrong with sunny_boundaries = {!r}".format(sunny_boundaries))

        boundary_surface_PB = plt3d.art3d.Poly3DCollection(corner_points_list, alpha=0.15)
        boundary_surface_PB.set_color("gray")
        boundary_surface_PB.set_edgecolor("gray")
        ax3d.add_collection3d(boundary_surface_PB)


def example():
    num_agents = 5
    key = jax.random.PRNGKey(0)

    env = AYS_Environment(reward_type=["PB", "PB", "PB"])

    obs, state, graph_states = env.reset(key)

    for step in range(200):
        key, key_reset, key_act, key_step = jax.random.split(key, 4)

        fig = env.render(graph_states)  # TODO is this the best way to do it?
        plt.savefig(f"project_name/images/{step}.png")
        plt.close()
        # print("obs:", obs)

        # Sample random actions.
        key_act = jax.random.split(key_act, env.num_agents)
        actions = {agent: env.action_space[agent].sample(key_act[i]) for i, agent in enumerate(env.agents)}
        actions = {agent: 0 for i, agent in enumerate(env.agents)}

        # print("action:", env.game_actions_idx[actions[env.agents[state.agent_in_room]].item()])

        # Perform the step transition.
        obs, state, reward, done, infos, graph_states = env.step(key_step, state, actions, graph_states)
        # print(state)
        #
        # print("reward:", reward["agent_0"])

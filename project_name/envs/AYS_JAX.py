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
                 plot_progress=False, num_agents=1):
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
        # self.observation_space = torch.tensor([0.5, 0.5, 0.5, 10.0 / 20]).repeat(self.num_agents, 1)  # TODO sort this out basos

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

        assert jnp.all(state[:, 0] == state[0, 0]), "A - First column values are not equal."

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

        return self._get_obs(env_state), wrapper_state

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
             ) -> Tuple[Dict[str, chex.Array], InfoState, Dict[str, float], Dict[str, bool], Dict]:
        """Performs step transitions in the environment."""

        key, key_reset = jax.random.split(key)
        obs_st, states_st, rewards, dones, infos = self.step_env(key, state, actions)

        # TODO need to add stop if episode runs on too long

        obs_re, states_re = self.reset(key_reset)

        # Auto-reset environment based on termination
        states = jax.tree_map(lambda x, y: jax.lax.select(dones["__all__"], x, y), states_re,
                              states_st)  # TODO need to understand this a little bit better
        obs = jax.tree_map(lambda x, y: jax.lax.select(dones["__all__"], x, y), obs_re, obs_st)

        # add info stuff idk if right spot
        def _batchify_floats(x: dict):
            return jnp.stack([x[a] for a in self.agents])

        ep_done = dones["__all__"]
        batch_reward = _batchify_floats(rewards)
        new_episode_return = state.episode_returns + _batchify_floats(rewards)
        new_episode_length = state.episode_lengths + 1
        new_won_episode = jnp.any(jnp.array([infos["agent_done_causation"][agent] for agent in self.agents]) == 1).astype(dtype=jnp.float32)  # TODO check this works - also make sure doesn't randomly say 1 and uses another win
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

        return obs, wrapper_state, rewards, dones, infos

    @partial(jax.jit, static_argnums=(0,))
    def step_env(self, key: chex.PRNGKey, state: InfoState, actions: dict) -> Tuple[
        Dict[str, chex.Array], InfoState, Dict[str, float], Dict[str, bool], Dict]:
        actions = jnp.array([actions[i] for i in self.agents])

        step = state.env_state.step + 1  # TODO should it be 1 or dt

        action_matrix = self._get_parameters(actions)

        traj_one_step = odeint(self._ays_rescaled_rhs_marl, state.env_state.ayse,
                               jnp.array([state.env_state.step, step], dtype=jnp.float32),
                               action_matrix, mxstep=50000)
        # TODO results match if it is using x64 bit precision but basically close with x32, worth mentioning in paper

        new_state = traj_one_step[1]
        assert jnp.all(new_state[:, 0] == new_state[0, 0]), "A - First column values are not equal."
        # TODO above works if don't call early stop stuff if you remember

        env_state = state.env_state
        env_state = env_state.replace(ayse=new_state,
                                      prev_actions=actions,
                                      step=step,
                                      terminal=~jnp.array(self._inside_planetary_boundaries_all(new_state)),
                                      # TODO check the tilde works
                                      )

        # convert state to obs
        obs = self._get_obs(env_state)

        # do the agent dones as well idk if it be useful for later on
        dones = {agent: jnp.array(~jnp.array(
            self._inside_planetary_boundaries(new_state, self.agent_ids[agent])) or self._arrived_at_final_state(
            new_state, self.agent_ids[agent])) for agent in self.agents}  # TODO check the tilde actually works

        # reward function innit
        rewards = self._get_rewards(env_state.ayse)

        # for agent in self.agents:  # TODO is this okay idk?
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
                info)

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

        adot = (E_tot - (A_matrix / args[:, 5])) * ays_inv_matrix[:, 0] * ays_inv_matrix[:, 0] / A_mid  # TODO check this maths
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
    def _get_rewards(self, ayse):
        def reward_distance_PB(agent):
            if self._inside_planetary_boundaries(ayse, agent):
                return jnp.linalg.norm(ayse[agent, :3] - self.PB_2)
            else:
                return 0.0

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
            rewards = rewards.at[agent_index].set(agent_reward)

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
    def _compactification(self, x, x_mid):
        if x == 0:
            return 0.
        if x == np.infty:
            return 1.
        return x / (x + x_mid)

    @partial(jax.jit, static_argnums=(0,))
    def _inv_compactification(self, y, x_mid):
        if y == 0:
            return 0.
        if np.allclose(y, 1):
            return np.infty
        return x_mid * y / (1 - y)

    @partial(jax.jit, static_argnums=(0,))
    def _inside_planetary_boundaries(self, ayse, agent_index):  # TODO can we subsume all into the individual somehow
        e = ayse[agent_index, 3]
        y = ayse[agent_index, 1]
        a = ayse[agent_index, 0]
        is_inside = True

        if a > self.A_PB[agent_index] or y < self.Y_PB[agent_index] or e > self.E_LIMIT[agent_index]:
            is_inside = False
        return is_inside

    @partial(jax.jit, static_argnums=(0,))
    def _inside_planetary_boundaries_all(self, ayse):
        e = ayse[:, 3]
        y = ayse[:, 1]
        a = ayse[:, 0]
        is_inside = True

        if jnp.all(a > self.A_PB) or jnp.all(y < self.Y_PB) or jnp.all(e > self.E_LIMIT):
            is_inside = False
        return is_inside

    @partial(jax.jit, static_argnums=(0,))
    def _arrived_at_final_state(self, ayse, agent_index):
        if jnp.any(jnp.abs(ayse - self.green_fp)[agent_index, :] < self.final_radius):  # TODO confirm this works
            return True
        elif jnp.any(jnp.abs(ayse - self.black_fp)[agent_index, :] < self.final_radius):
            return True
        else:
            return False

    @partial(jax.jit, static_argnums=(0,))
    def _green_fixed_point(self, ayse, agent_index):  # TODO can we add this to the above to save code space
        if jnp.any(jnp.abs(ayse - self.green_fp)[agent_index, :] < self.final_radius):  # TODO confirm this works
            return True
        else:
            return False

    @partial(jax.jit, static_argnums=(0,))
    def _black_fixed_point(self, ayse, agent_index):  # TODO can we add this to the above to save code space
        if jnp.any(jnp.abs(ayse - self.black_fp)[agent_index, :] < self.final_radius):  # TODO confirm this works
            return True
        else:
            return False

    # def _good_final_state(self, agent):
    #     e = self.state[agent, 0]
    #     y = self.state[agent, 1]
    #     a = self.state[agent, 2]
    #     if np.abs(e - self.green_fp[agent, 0]) < self.final_radius[agent] \
    #             and np.abs(y - self.green_fp[agent, 1]) < self.final_radius[agent] \
    #             and np.abs(a - self.green_fp[agent, 2]) < self.final_radius[agent]:
    #         return True
    #     else:
    #         return False

    @partial(jax.jit, static_argnums=(0,))
    def _which_final_state(self, ayse, agent_index):  # TODO same here can we add the green fixed point tINGG
        if self._green_fixed_point(ayse, agent_index):
            return 1
        elif self._black_fixed_point(ayse, agent_index):
            return 2
        else:
            return self._which_PB(ayse, agent_index)

    @partial(jax.jit, static_argnums=(0,))
    def _which_PB(self, ayse, agent_index):
        if ayse[agent_index, 0] >= self.A_PB:
            return 3
        elif ayse[agent_index, 1] <= self.Y_PB:
            return 4
        elif ayse[agent_index, 2] <= self.S_LIMIT:  # TODO check this ting
            return 5
        # elif ayse[agent_index, 3] >= self.E_LIMIT:  # TODO do we need an e limit idk?
        #     return 6
        else:
            return 7

    @partial(jax.jit, static_argnums=(0,))
    def _done_causation(self, state: EnvState):  # TODO check this works - it says black fixed point but this not always true, cus of the or function
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
            if state.dones[agent]:
                state.done_causation[agent] = self._which_final_state(state.ayse, self.agent_ids[agent])
            else:
                state.done_causation[agent] = 0

        return state


def example():
    num_agents = 5
    key = jax.random.PRNGKey(0)

    env = AYS_Environment(reward_type=["PB", "PB", "PB"])

    obs, state = env.reset(key)
    # env.render(state)

    for _ in range(2000):
        key, key_reset, key_act, key_step = jax.random.split(key, 4)

        # env.render(state)
        # print("obs:", obs)

        # Sample random actions.
        key_act = jax.random.split(key_act, env.num_agents)
        actions = {agent: env.action_space[agent].sample(key_act[i]) for i, agent in enumerate(env.agents)}

        # print("action:", env.game_actions_idx[actions[env.agents[state.agent_in_room]].item()])

        # Perform the step transition.
        obs, state, reward, done, infos = env.step(key_step, state, actions)
        # print(state)
        #
        print("reward:", reward["agent_0"])

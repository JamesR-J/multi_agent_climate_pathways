import jax
import jax.numpy as jnp
import jax.random as jrandom
from project_name.agents.agent_main import Agent
import sys
from ..utils import import_class_from_folder, batchify
from functools import partial
from typing import Any


class MultiAgent(Agent):
    def __init__(self, env, config, key):
        super().__init__(env, config, key)
        self.agent_list = {agent: None for agent in env.agents}  # TODO is there a better way to do this?
        self.hstate = jnp.zeros((env.num_agents, config['NUM_ENVS'], config["GRU_HIDDEN_DIM"]))
        self.train_state_list = {agent: None for agent in env.agents}
        for agent in env.agents:
            self.agent_list[agent] = (
                import_class_from_folder(self.agent_types[agent])(env=env, key=key, config=config))
            train_state, init_hstate = self.agent_list[agent].create_train_state()
            self.hstate = self.hstate.at[env.agent_ids[agent], :].set(init_hstate)
            self.train_state_list[agent] = train_state
            # TODO this is dodgy list train_state way, but can we make a custom sub class, is that faster?

    @partial(jax.jit, static_argnums=(0,))
    def initialise(self):
        return self.train_state_list, self.hstate

    @partial(jax.jit, static_argnums=(0,))
    def act(self, train_state: Any, hstate: Any, obs_batch: Any, last_done: Any, key: Any):  # TODO add better chex
        action_n = jnp.zeros((self.env.num_agents, self.config["NUM_ENVS"]), dtype=int)
        value_n = jnp.zeros((self.env.num_agents, self.config["NUM_ENVS"]))
        log_prob_n = jnp.zeros((self.env.num_agents, self.config["NUM_ENVS"]))
        for agent in self.env.agents:
            ac_in = (obs_batch[jnp.newaxis, self.env.agent_ids[agent], :],
                     last_done[jnp.newaxis, self.env.agent_ids[agent]],
                     )
            ind_hstate, ind_action, ind_log_prob, ind_value, key = self.agent_list[agent].act(train_state[agent],
                                                                                              hstate[
                                                                                              self.env.agent_ids[agent],
                                                                                              :],
                                                                                              ac_in, key)
            action_n = action_n.at[self.env.agent_ids[agent]].set(ind_action[0])
            value_n = value_n.at[self.env.agent_ids[agent]].set(ind_value[0])
            log_prob_n = log_prob_n.at[self.env.agent_ids[agent]].set(ind_log_prob[0])
            hstate = hstate.at[self.env.agent_ids[agent], :].set(ind_hstate[0])

        return hstate, action_n, log_prob_n, value_n, key

        # def _agent_act(agent_id, hstate, obs_batch, last_done, key):
        #     ac_in = (obs_batch[jnp.newaxis, :],
        #              last_done[jnp.newaxis],
        #              )
        #     # print(agent_list[env.agents[agent_id]])
        #     # print(agent_id)
        #     # # print(train_state_list)
        #     # sys.exit()
        #     hstate, action, log_prob, value = agent_list[env.agents[agent_id]].act(train_state_list[env.agents[agent_id]], hstate, ac_in, key)
        #     return hstate, action, log_prob, value
        #
        # reset_key = jrandom.split(key, env.num_agents)
        # # print(agent_list)
        # # print(batchify(train_state_list, env.agents))
        # # print(hstate)
        # # print(obs_batch)
        # # print(last_done)
        # # print(reset_key)
        # hstate, action_n, log_prob_n, value_n, key = jax.vmap(_agent_act)(batchify(env.agent_ids, env.agents)[:, jnp.newaxis], hstate, obs_batch, last_done, reset_key)
        # sys.exit()

    @partial(jax.jit, static_argnums=(0,))
    def update(self, update_state: Any, trajectory_batch: Any):  # TODO add better chex
        train_state, env_state, last_obs, last_done, hstate, key = update_state
        last_obs_batch = batchify(last_obs, self.env.agents, self.env.num_agents, self.config["NUM_ENVS"])
        for agent in self.env.agents:  # TODO this is probs mega slowsies
            # TODO be good to pmap agents as think this is the biggest storage draw, i,e. distribute agents between GPU
            individual_trajectory_batch = jax.tree_map(lambda x: x[:, self.env.agent_ids[agent]], trajectory_batch)
            individual_train_state = (train_state[agent], env_state, last_obs_batch[self.env.agent_ids[agent], :],
                                      last_done[self.env.agent_ids[agent]],
                                      hstate[self.env.agent_ids[agent], :], key)
            individual_runner_list = self.agent_list[agent].update(individual_train_state, individual_trajectory_batch)
            train_state[agent] = individual_runner_list[0]
            hstate = hstate.at[self.env.agent_ids[agent], :].set(individual_runner_list[4])
            key = individual_runner_list[-1]

        return train_state, env_state, last_obs, last_done, hstate, key

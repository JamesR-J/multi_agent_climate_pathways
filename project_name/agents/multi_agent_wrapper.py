import chex
import jax
import jax.numpy as jnp
from project_name.agents.agent_main import Agent
from project_name.utils import import_class_from_folder, batchify
from functools import partial
from typing import Any, Tuple
from flax.training.train_state import TrainState


class MultiAgent(Agent):
    def __init__(self, env: Any, config: dict, key: chex.PRNGKey):
        super().__init__(env, config, key)
        self.agent_list = {agent: None for agent in env.agents}
        self.hstate = jnp.zeros((env.num_agents, config['NUM_ENVS'], config["GRU_HIDDEN_DIM"]))
        self.train_state_dict = {agent: None for agent in env.agents}
        for agent in env.agents:
            self.agent_list[agent] = (
                import_class_from_folder(self.agent_types[agent])(env=env, key=key, config=config))
            train_state, init_hstate = self.agent_list[agent].create_train_state()
            self.hstate = self.hstate.at[env.agent_ids[agent], :].set(init_hstate)
            self.train_state_dict[agent] = train_state

    @partial(jax.jit, static_argnums=(0,))
    def initialise(self) -> Tuple[dict, chex.Array]:
        return self.train_state_dict, self.hstate

    @partial(jax.jit, static_argnums=(0,))
    def act(self, train_state: TrainState, hstate: chex.Array, obs_batch: chex.Array, last_done: chex.Array,
            key: chex.PRNGKey) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.PRNGKey, chex.Array,
    chex.PRNGKey]:
        action_n = jnp.zeros((self.env.num_agents, self.config["NUM_ENVS"]), dtype=int)
        value_n = jnp.zeros((self.env.num_agents, self.config["NUM_ENVS"]))
        log_prob_n = jnp.zeros((self.env.num_agents, self.config["NUM_ENVS"]))
        pi_n = jnp.zeros((self.env.num_agents, self.config["NUM_ENVS"], self.env.action_space(self.env.agents[0]).n))
        spec_key_n = jnp.zeros((self.env.num_agents, 2))
        for agent in self.env.agents:
            ac_in = (obs_batch[jnp.newaxis, self.env.agent_ids[agent], :],
                     last_done[jnp.newaxis, self.env.agent_ids[agent]],
                     )
            ind_hstate, ind_action, ind_log_prob, ind_value, key, pi, spec_key = (
                self.agent_list[agent].act(train_state[agent], hstate[self.env.agent_ids[agent], :], ac_in, key))
            action_n = action_n.at[self.env.agent_ids[agent]].set(ind_action[0])
            value_n = value_n.at[self.env.agent_ids[agent]].set(ind_value[0])
            log_prob_n = log_prob_n.at[self.env.agent_ids[agent]].set(ind_log_prob[0])
            pi_n = pi_n.at[self.env.agent_ids[agent]].set(pi[0])
            spec_key_n = spec_key_n.at[self.env.agent_ids[agent]].set(spec_key)
            hstate = hstate.at[self.env.agent_ids[agent], :].set(ind_hstate[0])

        return hstate, action_n, log_prob_n, value_n, key, pi_n, spec_key_n

    @partial(jax.jit, static_argnums=(0,))
    def update(self, update_state: chex.Array, trajectory_batch: chex.Array) -> Tuple[TrainState, chex.Array, chex.Array, chex.Array, chex.Array, chex.PRNGKey]:
        train_state, env_state, last_obs, last_done, hstate, key = update_state
        last_obs_batch = batchify(last_obs, self.env.agents, self.env.num_agents, self.config["NUM_ENVS"])
        for agent in self.env.agents:
            individual_trajectory_batch = jax.tree_map(lambda x: x[:, self.env.agent_ids[agent]], trajectory_batch)
            individual_train_state = (train_state[agent], env_state, last_obs_batch[self.env.agent_ids[agent], :],
                                      last_done[self.env.agent_ids[agent]],
                                      hstate[self.env.agent_ids[agent], :], key)
            individual_runner_list = self.agent_list[agent].update(individual_train_state, individual_trajectory_batch)
            train_state[agent] = individual_runner_list[0]
            hstate = hstate.at[self.env.agent_ids[agent], :].set(individual_runner_list[4])
            key = individual_runner_list[-1]

        return train_state, env_state, last_obs, last_done, hstate, key

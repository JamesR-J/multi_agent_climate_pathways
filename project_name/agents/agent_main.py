import jax
import jax.numpy as jnp
import jax.random as jrandom
from ..utils import import_class_from_folder
from functools import partial
from typing import Any
import chex
import sys


# initialise agents from the config file deciding what the algorithms are
class Agent:
    def __init__(self, env, config, key):  # TODO add better chex
        self.env = env
        self.agent_types = {agent: config["AGENT_TYPE"][env.agent_ids[agent]] for agent in env.agents}
        self.agent = import_class_from_folder(self.agent_types[env.agents[env.num_agents-1]])(env=env, key=key, config=config)
        # TODO some conditional if its env.num_agents > 1 then it calls the multi-agent wrapper

    @partial(jax.jit, static_argnums=(0,))
    def initialise(self):
        train_state, hstate = self.agent.create_train_state()  # TODO add thing to return empty hstate if hstate is none I guess
        return train_state, hstate[jnp.newaxis, :]

    @partial(jax.jit, static_argnums=(0,))
    def act(self, train_state: Any, hstate: Any, obs_batch: Any, last_done: Any, key: Any):  # TODO add better chex
        ac_in = (obs_batch,
                 last_done,
                 )
        hstate, action, log_prob, value, key = self.agent.act(train_state, hstate[0], ac_in, key)  # squeeze extra ac_in dim
        return hstate[jnp.newaxis, :], action, log_prob, value, key

    @partial(jax.jit, static_argnums=(0,))
    def update(self, update_state: Any, trajectory_batch: Any):  # TODO add better chex
        train_state, env_state, obs, done_batch, hstate, key = update_state
        update_state = train_state, env_state, obs[self.env.agents[0]], done_batch.squeeze(axis=0), hstate[0], key
        return self.agent.update(update_state, trajectory_batch)

    # TODO add wrapper for multi agent on top of this single agent
    # TODO add a thing so hstate is fine if don't have one etc

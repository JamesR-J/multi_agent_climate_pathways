import jax
import jax.numpy as jnp
from project_name.utils import import_class_from_folder, batchify
from functools import partial
from typing import Any, Dict, Tuple
import chex
from flax.training.train_state import TrainState


# initialise agents from the config file deciding what the algorithms are
class Agent:
    def __init__(self, env: Any, config: dict, key: chex.PRNGKey):
        self.env = env
        self.config = config
        self.agent_types = {agent: config["AGENT_TYPE"][env.agent_ids[agent]] for agent in env.agents}
        self.agent = import_class_from_folder(self.agent_types[env.agents[env.num_agents - 1]])(env=env, key=key,
                                                                                                config=config)

    @partial(jax.jit, static_argnums=(0,))
    def initialise(self) -> Tuple[TrainState, chex.Array]:
        train_state, hstate = self.agent.create_train_state()
        return train_state, hstate[jnp.newaxis, :]

    @partial(jax.jit, static_argnums=(0,))
    def act(self, train_state: TrainState, hstate: chex.Array, obs_batch: Dict[str, chex.Array], last_done: chex.Array,
            key: chex.PRNGKey) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.PRNGKey, chex.Array,
    chex.PRNGKey]:
        ac_in = (obs_batch,
                 last_done,
                 )
        hstate, action, log_prob, value, key, pi, spec_key = self.agent.act(train_state, hstate[0], ac_in, key)
        return hstate[jnp.newaxis, :], action, log_prob, value, key, pi, spec_key[jnp.newaxis, :]

    @partial(jax.jit, static_argnums=(0,))
    def update(self, update_state: chex.Array, trajectory_batch: chex.Array) -> Tuple[Any, Any, Dict[str, chex.Array],
    chex.Array, chex.Array, chex.PRNGKey]:
        train_state, env_state, last_obs, done_batch, hstate, key = update_state
        last_obs_batch = batchify(last_obs, self.env.agents, self.env.num_agents, self.config["NUM_ENVS"])
        update_state = train_state, env_state, last_obs_batch[0], done_batch.squeeze(axis=0), hstate[0], key
        trajectory_batch = jax.tree_map(lambda x: x[:, 0], trajectory_batch)
        train_state, env_state, obs, last_done, hstate, key = self.agent.update(update_state, trajectory_batch)
        return train_state, env_state, last_obs, last_done[jnp.newaxis, :], hstate[jnp.newaxis], key

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
import yaml
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
import distrax
import jaxmarl
from jaxmarl.wrappers.baselines import MPELogWrapper as LogWrapper
import matplotlib.pyplot as plt
import hydra
from omegaconf import OmegaConf
import wandb
from .envs.AYS_JAX import AYS_Environment
import sys



class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def batchify(x: dict, agent_list, num_actors):
    max_dim = max([x[a].shape[-1] for a in agent_list])

    def pad(z, length):
        return jnp.concatenate([z, jnp.zeros(z.shape[:-1] + [length - z.shape[-1]])], -1)

    x = jnp.stack([x[a] if x[a].shape[-1] == max_dim else pad(x[a]) for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def make_eval(config, orbax_checkpointer):
    """
    run pure evaluation no updates
    no batch envs
    be good for option to render as well so avoid jit if possible
    but still jit if we want just number of eval steps and some metrics basos
    """

    env = AYS_Environment(reward_type=config["REWARD_TYPE"], num_agents=config["NUM_AGENTS"])
    config["NUM_ENVS"] = 1  # TODO can we adjust num envs so we can get rendering or not?
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
            config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
            config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac

    def train(rng):
        # INIT NETWORK
        network = ActorCritic(env.action_space[env.agents[0]].n, activation=config["ACTIVATION"])
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space[env.agents[0]].shape)
        network_params = network.init(_rng, init_x)
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), optax.adam(config["LR"], eps=1e-5))

        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        target = {'model': train_state}  # must match the input dict
        train_state = orbax_checkpointer.restore('/tmp/flax_ckpt/orbax/single_save', item=target)["model"]

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obs, env_state, graph_state = jax.vmap(env.reset)(reset_rng)

        def _env_step(runner_state):
            train_state, env_state, last_obs, env_graph_state, rng = runner_state

            obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
            # SELECT ACTION
            rng, _rng = jax.random.split(rng)

            pi, value = network.apply(train_state.params, obs_batch)
            action = pi.sample(seed=_rng)
            log_prob = pi.log_prob(action)
            env_act = unbatchify(action, env.agents, config["NUM_ENVS"], env.num_agents)

            # STEP ENV
            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, config["NUM_ENVS"])
            obsv, env_state, reward, done, info, env_graph_state = jax.vmap(env.step)(rng_step, env_state, env_act, env_graph_state)
            # print(env_state)

            # info = jax.tree_map(lambda x: x.reshape((config["NUM_ACTORS"])), info)  # it used to be done by actors but now each env has the number of agent causations
            transition = Transition(
                batchify(done, env.agents, config["NUM_ACTORS"]).squeeze(),
                action,
                value,
                batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                log_prob,
                obs_batch,
                info,
            )

            env.render(env_graph_state[0])

            runner_state = (train_state, env_state, obsv, env_graph_state, rng)
            return runner_state, transition

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obs, graph_state, _rng)
        runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, config["NUM_EVAL_STEPS"])
        metric = traj_batch.info

        def callback(metric):
            wandb.log({
                    # the metrics have an agent dimension, but this is identical
                    # for all agents so index into the 0th item of that dimension.
                    "returns": metric["returned_episode_returns"][:, :, 0][
                        metric["returned_episode"][:, :, 0]
                    ].mean(),
                    "win_rate": metric["returned_won_episode"][:, :, 0][
                        metric["returned_episode"][:, :, 0]
                    ].mean(),
                    "env_step": metric["update_steps"]
                    * config["NUM_ENVS"]
                    * config["NUM_STEPS"],
                }
            )

        metric["update_steps"] = 0
        jax.experimental.io_callback(callback, None, metric)
        return {"runner_state": runner_state, "metrics": metric}

    return train

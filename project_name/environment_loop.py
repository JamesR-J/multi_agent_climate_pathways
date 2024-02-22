import sys
import jax
from jaxmarl.environments.smax import map_name_to_scenario, HeuristicEnemySMAX
from jaxmarl.wrappers.baselines import SMAXLogWrapper
import jax.numpy as jnp
from typing import Sequence, NamedTuple, Any, Dict
import wandb
import jax.random as jrandom
from .envs.AYS_JAX import AYS_Environment
from .agents.agent_main import Agent


class Transition(NamedTuple):
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def batchify(x: dict, agent_list):
    return jnp.stack([x[a] for a in agent_list])


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    return {a: x[i] for i, a in enumerate(agent_list)}


def run_train(config):
    env = AYS_Environment(reward_type=config["REWARD_TYPE"], num_agents=config["NUM_AGENTS"])
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"])
    config["MINIBATCH_SIZE"] = (config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"])
    config["CLIP_EPS"] = (config["CLIP_EPS"] / env.num_agents if config["SCALE_CLIP_EPS"] else config["CLIP_EPS"])

    key = jrandom.PRNGKey(config["SEED"])
    actor = Agent(env=env, config=config, key=key)
    train_state, hstate = actor.initialise()

    reset_key = jrandom.split(key, config["NUM_ENVS"])
    obs, env_state, graph_state = jax.vmap(env.reset, in_axes=(0,))(reset_key)
    runner_state = (train_state, env_state, obs, jnp.zeros((env.num_agents, config["NUM_ENVS"]), dtype=bool), hstate, graph_state, key)

    def _run_update(update_runner_state, unused):
        runner_state, update_steps = update_runner_state

        def _run_episode_step(runner_state, unused):
            # take initial env_state
            train_state, env_state, last_obs, last_done, hstate, env_graph_state, key = runner_state

            # act on this initial env_state
            obs_batch = batchify(last_obs, env.agents)
            action_n = jnp.zeros((env.num_agents, config["NUM_ENVS"]), dtype=int)
            value_n = jnp.zeros((env.num_agents, config["NUM_ENVS"]))
            log_prob_n = jnp.zeros((env.num_agents, config["NUM_ENVS"]))


            hstate, action, log_prob, value, key = actor.act(train_state, hstate, obs_batch, last_done, key)
            env_act = unbatchify(action_n, env.agents, config["NUM_ENVS"], env.num_agents)
            env_act = {k: v.squeeze() for k, v in env_act.items()}

            # step in env
            key, _key = jrandom.split(key)
            key_step = jrandom.split(_key, config["NUM_ENVS"])
            obs, env_state, reward, done, info, env_graph_state = jax.vmap(env.step, in_axes=(0, 0, 0, 0))(key_step, env_state, env_act, env_graph_state)

            # update tings my dude
            # swaps agent id axis and envs so that agent id comes first
            info = jax.tree_map(lambda x: jnp.swapaxes(x, 0, 1), info)
            done_batch = batchify(done, env.agents)
            transition = Transition(jnp.full((env.num_agents, config["NUM_ENVS"]), done["__all__"]),
                                    done_batch,
                                    action_n,
                                    value_n,
                                    batchify(reward, env.agents),
                                    log_prob_n,
                                    obs_batch,
                                    info,
                                    )

            return (train_state, env_state, obs, done_batch, hstate, env_graph_state, key), transition

        # run for NUM_STEPS length rollout
        runner_state, trajectory_batch = jax.lax.scan(_run_episode_step, runner_state, None, config["NUM_STEPS"])
        train_state, env_state, obs, done_batch, hstate, env_graph_state, key = runner_state

        # update agents here after rollout
        update_state = train_state, env_state, obs, done_batch, hstate, key
        update_state = actor.update(update_state, trajectory_batch)
        train_state = update_state[0]
        last_obs = update_state[2]
        last_done = update_state[3]
        key = update_state[-1]  # TODO make this name rather than number index would probs be good inni, same as above

        # metric handling
        metric = jax.tree_map(lambda x: x.reshape((config["NUM_STEPS"], config["NUM_ENVS"], env.num_agents)),
                              trajectory_batch.info)
        # TODO check the above if add multiple envs innit

        def callback(metric):
            wandb.log({
                # the metrics have an agent dimension, but this is identical
                # for all agents so index into the 0th item of that dimension. not true anymore as hetero babY
                "returns": metric["returned_episode_returns"][:, :, 0][metric["returned_episode"][:, :, 0]].mean(),  # TODO this is fine for all coop but need to add extra dims for multiagent
                "win_rate": metric["returned_won_episode"][:, :, 0][metric["returned_episode"][:, :, 0]].mean(),
                "env_step": metric["update_steps"] * config["NUM_ENVS"] * config["NUM_STEPS"]})

        metric["update_steps"] = update_steps
        jax.experimental.io_callback(callback, None, metric)
        update_steps = update_steps + 1

        return ((train_state, env_state, last_obs, last_done, hstate, graph_state, key), update_steps), metric

    runner_state, metric = jax.lax.scan(_run_update, (runner_state, 0), None, config["NUM_UPDATES"])

    return {"runner_state": runner_state, "metrics": metric}


def run_eval(config):
    pass



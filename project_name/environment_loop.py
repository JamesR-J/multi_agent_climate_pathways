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
from .agents.multi_agent_wrapper import MultiAgent
from .utils import batchify, unbatchify, Transition


def run_train(config):
    env = AYS_Environment(reward_type=config["REWARD_TYPE"], num_agents=config["NUM_AGENTS"], homogeneous=config["HOMOGENEOUS"])
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"])
    config["MINIBATCH_SIZE"] = (config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"])
    config["CLIP_EPS"] = (config["CLIP_EPS"] / env.num_agents if config["SCALE_CLIP_EPS"] else config["CLIP_EPS"])

    def train():
        key = jrandom.PRNGKey(config["SEED"])

        config["NUM_DEVICES"] = len(jax.local_devices())

        if env.num_agents == 1:
            actor = Agent(env=env, config=config, key=key)
        else:
            actor = MultiAgent(env=env, config=config, key=key)
        train_state, hstate = actor.initialise()

        # reset_key = jrandom.split(key, config["NUM_ENVS"])
        reset_key = jrandom.split(key, config["NUM_ENVS"]).reshape(config["NUM_DEVICES"], config["NUM_ENVS"] // config["NUM_DEVICES"], -1)

        # obs, env_state, graph_state = jax.vmap(env.reset, in_axes=(0,))(reset_key)
        vreset = jax.jit(jax.vmap(env.reset, in_axes=(0,), out_axes=(0, 0, 0), axis_name="batch_axis"))
        obs, env_state, graph_state = jax.pmap(vreset, out_axes=(0, 0, 0), axis_name="device_axis")(reset_key)

        runner_state = (train_state, env_state, obs, jnp.zeros((env.num_agents, config["NUM_ENVS"]), dtype=bool), hstate, graph_state, key)

        def _run_update(update_runner_state, unused):
            runner_state, update_steps = update_runner_state

            def _run_episode_step(runner_state, unused):
                # take initial env_state
                train_state, env_state, last_obs, last_done, hstate, env_graph_state, key = runner_state

                # act on this initial env_state
                obs_batch = batchify(last_obs, env.agents, env.num_agents, config["NUM_ENVS"])
                hstate, action_n, log_prob_n, value_n, key = actor.act(train_state, hstate, obs_batch, last_done, key)
                env_act = unbatchify(action_n, env.agents, env.num_agents, config["NUM_DEVICES"])
                env_act = {k: v for k, v in env_act.items()}  # TODO chck the axis squeeze v.squeeze(axis=1)

                # step in env
                key, _key = jrandom.split(key)
                # key_step = jrandom.split(_key, config["NUM_ENVS"])
                key_step = jrandom.split(key, config["NUM_ENVS"]).reshape(config["NUM_DEVICES"],
                                                                           config["NUM_ENVS"] // config["NUM_DEVICES"],
                                                                           -1)
                # obs, env_state, reward, done, info, env_graph_state = jax.vmap(env.step, in_axes=(0, 0, 0, 0))(key_step, env_state, env_act, env_graph_state)
                vstep = jax.vmap(env.step, in_axes=(0, 0, 0, 0), axis_name="batch_axis")
                obs, env_state, reward, done, info, env_graph_state = jax.pmap(vstep, out_axes=(0, 0, 0, 0, 0, 0), axis_name="device_axis")(key_step, env_state, env_act, env_graph_state)

                # update tings my dude
                # swaps agent id axis and envs so that agent id comes first
                info = jax.tree_map(lambda x: jnp.swapaxes(x.reshape(config["NUM_ENVS"], -1), -0, 1), info)
                done_batch = batchify(done, env.agents, env.num_agents, config["NUM_ENVS"]).squeeze(axis=2)
                transition = Transition(jnp.full((env.num_agents, config["NUM_ENVS"]), done["__all__"].reshape((config["NUM_ENVS"]))),
                                        done_batch,
                                        action_n,
                                        value_n,
                                        batchify(reward, env.agents, env.num_agents, config["NUM_ENVS"]).squeeze(axis=2),
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

            def callback(metric):
                wandb.log({
                    # the metrics have an agent dimension, but this is identical
                    # for all agents so index into the 0th item of that dimension. not true anymore as hetero babY
                    "returns": metric["returned_episode_returns"][:, :, 0][metric["returned_episode"][:, :, 0]].mean(),  # TODO this is fine for all coop but need to add extra dims for multiagent
                    "win_rate": metric["returned_won_episode"][:, :, 0][metric["returned_episode"][:, :, 0]].mean(),  # TODO removed last 0 index making it all averaged
                    "env_step": metric["update_steps"] * config["NUM_ENVS"] * config["NUM_STEPS"]})

            metric["update_steps"] = update_steps
            jax.experimental.io_callback(callback, None, metric)
            update_steps = update_steps + 1

            return ((train_state, env_state, last_obs, last_done, hstate, graph_state, key), update_steps), metric

        runner_state, metric = jax.lax.scan(_run_update, (runner_state, 0), None, config["NUM_UPDATES"])

        return {"runner_state": runner_state, "metrics": metric}

    return train


# def run_eval(config, orbax_checkpointer, chkpt_save_path):
#     # TODO lots to update here
#     # TODO can we reduce reusing code from above, would we make this a class? or have to import things
#     # TODO or export config from the above? idk and the env?
#     env = AYS_Environment(reward_type=config["REWARD_TYPE"], num_agents=config["NUM_AGENTS"])
#     config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]  # TODO do we want to hard define 1 env?
#     config["NUM_UPDATES"] = (config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"])
#     config["MINIBATCH_SIZE"] = (config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"])
#     config["CLIP_EPS"] = (config["CLIP_EPS"] / env.num_agents if config["SCALE_CLIP_EPS"] else config["CLIP_EPS"])
#
#     key = jrandom.PRNGKey(config["SEED"])
#     if env.num_agents == 1:
#         actor = Agent(env=env, config=config, key=key)
#     else:
#         actor = MultiAgent(env=env, config=config, key=key)
#     train_state, hstate = actor.initialise()
#
#     target = {'model': train_state}  # must match the input dict
#     train_state = orbax_checkpointer.restore(chkpt_save_path, item=target)["model"]
#     # TODO need to adjust above if gonna be using multiagent or single agent
#
#     reset_key = jrandom.split(key, config["NUM_ENVS"])
#     obs, env_state, graph_state = jax.vmap(env.reset, in_axes=(0,))(reset_key)
#     runner_state = (train_state, env_state, obs, jnp.zeros((env.num_agents, config["NUM_ENVS"]), dtype=bool), hstate, graph_state, key)
#
#     def _eval_step(runner_state, unused):
#         # take initial env_state
#         train_state, env_state, last_obs, last_done, hstate, env_graph_state, key = runner_state
#
#         # act on this initial env_state
#         obs_batch = batchify(last_obs, env.agents)
#         hstate, action_n, log_prob_n, value_n, key = actor.act(train_state, hstate, obs_batch, last_done, key)
#         env_act = unbatchify(action_n, env.agents, config["NUM_ENVS"], env.num_agents)
#         env_act = {k: v.squeeze() for k, v in env_act.items()}
#
#         # step in env
#         key, _key = jrandom.split(key)
#         key_step = jrandom.split(_key, config["NUM_ENVS"])
#         obs, env_state, reward, done, info, env_graph_state = jax.vmap(env.step, in_axes=(0, 0, 0, 0))(key_step,
#                                                                                                        env_state,
#                                                                                                        env_act,
#                                                                                                        env_graph_state)
#
#         # update tings my dude
#         # swaps agent id axis and envs so that agent id comes first
#         info = jax.tree_map(lambda x: jnp.swapaxes(x, 0, 1), info)
#         done_batch = batchify(done, env.agents)
#         transition = Transition(jnp.full((env.num_agents, config["NUM_ENVS"]), done["__all__"]),
#                                 done_batch,
#                                 action_n,
#                                 value_n,
#                                 batchify(reward, env.agents),
#                                 log_prob_n,
#                                 obs_batch,
#                                 info,
#                                 )
#
#         env.render(env_graph_state[0])
#
#         return (train_state, env_state, obs, done_batch, hstate, env_graph_state, key), transition
#
#     # run for NUM_STEPS length rollout
#     runner_state, trajectory_batch = jax.lax.scan(_eval_step, runner_state, None, config["NUM_EVAL_STEPS"])
#
#     # metric handling
#     metric = jax.tree_map(lambda x: x.reshape((config["NUM_STEPS"], config["NUM_ENVS"], env.num_agents)),
#                           trajectory_batch.info)
#
#     # TODO check the above if add multiple envs innit
#
#     def callback(metric):
#         wandb.log({
#             # the metrics have an agent dimension, but this is identical
#             # for all agents so index into the 0th item of that dimension. not true anymore as hetero babY
#             "returns": metric["returned_episode_returns"][:, :, 0][metric["returned_episode"][:, :, 0]].mean(),
#             # TODO this is fine for all coop but need to add extra dims for multiagent
#             "win_rate": metric["returned_won_episode"][:, :, 0][metric["returned_episode"][:, :, 0]].mean(),
#             "env_step": metric["update_steps"] * config["NUM_ENVS"] * config["NUM_STEPS"]})
#
#     metric["update_steps"] = 0
#     jax.experimental.io_callback(callback, None, metric)
#
#     return {"runner_state": runner_state, "metrics": metric}



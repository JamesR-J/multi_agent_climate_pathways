import sys
import jax
from jaxmarl.environments.smax import map_name_to_scenario, HeuristicEnemySMAX
from jaxmarl.wrappers.baselines import SMAXLogWrapper
import jax.numpy as jnp
from typing import Sequence, NamedTuple, Any, Dict
import wandb
from .utils import import_class_from_folder
import jax.random as jrandom
from .envs.AYS_JAX import AYS_Environment


class Transition(NamedTuple):
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    # avail_actions: jnp.ndarray


def batchify(x: dict, agent_list):
    return jnp.stack([x[a] for a in agent_list])


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    return {a: x[i] for i, a in enumerate(agent_list)}


def run_train(config):
    # run a loop of episodes and stuff
    env = AYS_Environment(reward_type=config["REWARD_TYPE"], num_agents=config["NUM_AGENTS"])
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"])
    config["MINIBATCH_SIZE"] = (config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"])
    config["CLIP_EPS"] = (config["CLIP_EPS"] / env.num_agents if config["SCALE_CLIP_EPS"] else config["CLIP_EPS"])

    # initialise agents from the config file deciding what the algorithms are
    key = jrandom.PRNGKey(config["SEED"])
    agent_list = {agent: None for agent in env.agents}  # TODO is there a better way to do this?
    agent_types = {agent: config["AGENT_TYPE"][env.agent_ids[agent]] for agent in env.agents}
    hstate = jnp.zeros((env.num_agents, config['NUM_ENVS'], config["GRU_HIDDEN_DIM"]))
    train_state_list = {agent: None for agent in env.agents}
    for agent in env.agents:
        agent_list[agent] = (import_class_from_folder(agent_types[agent])(env=env, key=key, config=config))
        train_state, init_hstate = agent_list[agent].create_train_state()
        hstate = hstate.at[env.agent_ids[agent], :].set(init_hstate)
        train_state_list[agent] = train_state
        # TODO this is dodgy list train_state way, but can we make a custom sub class, is that faster?

    reset_key = jrandom.split(key, config["NUM_ENVS"])
    obs, env_state, graph_state = jax.vmap(env.reset, in_axes=(0,))(reset_key)
    """ train_state, env_state, obs, dones, hstate, key """  # TODO the runner_state setup
    runner_state = (train_state_list, env_state, obs, jnp.zeros((env.num_agents, config["NUM_ENVS"]), dtype=bool), hstate, graph_state, key)

    def _run_update(update_runner_state, unused):
        runner_state, update_steps = update_runner_state

        def _run_episode_step(runner_state, unused):
            # take initial env_state
            train_state_list, env_state, last_obs, last_done, hstate, env_graph_state, key = runner_state

            # act on this initial env_state
            obs_batch = batchify(last_obs, env.agents)
            action_n = jnp.zeros((env.num_agents, config["NUM_ENVS"]), dtype=int)
            value_n = jnp.zeros((env.num_agents, config["NUM_ENVS"]))
            log_prob_n = jnp.zeros((env.num_agents, config["NUM_ENVS"]))

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

            for agent in env.agents:
                ac_in = (obs_batch[jnp.newaxis, env.agent_ids[agent], :],
                         last_done[jnp.newaxis, env.agent_ids[agent]],
                         )
                ind_hstate, ind_action, ind_log_prob, ind_value, key = agent_list[agent].act(train_state_list[agent], hstate[env.agent_ids[agent], :], ac_in, key)
                action_n = action_n.at[env.agent_ids[agent]].set(ind_action[0])
                value_n = value_n.at[env.agent_ids[agent]].set(ind_value[0])
                log_prob_n = log_prob_n.at[env.agent_ids[agent]].set(ind_log_prob[0])
                hstate = hstate.at[env.agent_ids[agent], :].set(ind_hstate[0])
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

            runner_state = (train_state_list, env_state, obs, done_batch, hstate, env_graph_state, key)

            return runner_state, transition

        # run for NUM_STEPS length rollout
        runner_state, trajectory_batch = jax.lax.scan(_run_episode_step, runner_state, None, config["NUM_STEPS"])

        # update agents here after rollout
        train_state_list, env_state, last_obs, last_done, hstate, graph_state, key = runner_state
        for agent in env.agents:  # TODO this is probs mega slowsies
            last_obs_batch = batchify(last_obs, env.agents)
            individual_trajectory_batch = jax.tree_map(lambda x: x[:, env.agent_ids[agent]], trajectory_batch)
            individual_train_state = (train_state_list[agent], env_state, last_obs_batch[env.agent_ids[agent], :], last_done[env.agent_ids[agent]], hstate[env.agent_ids[agent], :], key)
            individual_runner_list = agent_list[agent].update(individual_train_state, individual_trajectory_batch)
            train_state_list[agent] = individual_runner_list[0]
            hstate = hstate.at[env.agent_ids[agent], :].set(individual_runner_list[4])
            key = individual_runner_list[-1]  # TODO make this name rather than number index would probs be good inni
        runner_state = (train_state_list, env_state, last_obs, last_done, hstate, graph_state, key)  # TODO is this unneccsary, can I maybe just call in place?

        # metric handling
        # metric = trajectory_batch.info
        metric = jax.tree_map(lambda x: x.reshape((config["NUM_STEPS"], config["NUM_ENVS"], env.num_agents)),
                              trajectory_batch.info)  # TODO check this if add multiple envs inni

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

        return (runner_state, update_steps), metric

    runner_state, metric = jax.lax.scan(_run_update, (runner_state, 0), None, config["NUM_UPDATES"])

    return {"runner_state": runner_state, "metrics": metric}


def run_eval(config):
    pass



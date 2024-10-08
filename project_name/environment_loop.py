"""
Adapted from JaxMARL
https://github.com/FLAIROx/JaxMARL/tree/main
"""

import jax
import jax.numpy as jnp
import wandb
import jax.random as jrandom
from .envs.AYS_JAX import AYS_Environment
from .agents.agent_main import Agent
from .agents.multi_agent_wrapper import MultiAgent
from .utils import batchify, unbatchify, Transition, EvalTransition
import matplotlib.pyplot as plt
import seaborn as sns
from .envs.graph_functions import create_figure_ays
import numpy as np


def run_train(config, checkpoint_manager, env_step_count_init=0, train_state_input=None):
    env = AYS_Environment(reward_type=config["REWARD_TYPE"],
                          num_agents=config["NUM_AGENTS"],
                          homogeneous=config["HOMOGENEOUS"],
                          climate_damages=config["CLIMATE_DAMAGES"])
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"])
    config["TOTAL_UPDATES"] = config["NUM_UPDATES"] * config["NUM_LOOPS"]
    config["MINIBATCH_SIZE"] = (config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"])
    config["CLIP_EPS"] = (config["CLIP_EPS"] / env.num_agents if config["SCALE_CLIP_EPS"] else config["CLIP_EPS"])

    def train():
        key = jrandom.PRNGKey(config["SEED"])

        if env.num_agents == 1:
            actor = Agent(env=env, config=config, key=key)
        else:
            actor = MultiAgent(env=env, config=config, key=key)
        train_state, hstate = actor.initialise()

        if train_state_input is not None:
            train_state = train_state_input

        reset_key = jrandom.split(key, config["NUM_ENVS"]).reshape(config["NUM_DEVICES"],
                                                                   config["NUM_ENVS"] // config["NUM_DEVICES"], -1)

        vreset = jax.jit(jax.vmap(env.reset, in_axes=(0,), out_axes=(0, 0, 0), axis_name="batch_axis"))
        obs, env_state, graph_state = jax.pmap(vreset, out_axes=(0, 0, 0), axis_name="device_axis")(reset_key)

        runner_state = (train_state, env_state, obs,
                        jnp.zeros((env.num_agents, config["NUM_ENVS"]), dtype=bool), hstate, graph_state, key)

        def _run_update(update_runner_state, unused):
            runner_state, update_steps = update_runner_state

            def _run_episode_step(runner_state, unused):
                # take initial env_state
                train_state, env_state, last_obs, last_done, hstate, env_graph_state, key = runner_state

                # act on this initial env_state
                obs_batch = batchify(last_obs, env.agents, env.num_agents, config["NUM_ENVS"])
                hstate, action_n, log_prob_n, value_n, key, _, _ = actor.act(train_state, hstate, obs_batch, last_done, key)
                env_act = unbatchify(action_n, env.agents, env.num_agents, config["NUM_DEVICES"])
                env_act = {k: v for k, v in env_act.items()}

                # step in env
                key, _key = jrandom.split(key)
                key_step = jrandom.split(key, config["NUM_ENVS"]).reshape(config["NUM_DEVICES"],
                                                                          config["NUM_ENVS"] // config["NUM_DEVICES"],
                                                                          -1)
                vstep = jax.vmap(env.step, in_axes=(0, 0, 0, 0), axis_name="batch_axis")
                obs, env_state, reward, done, info, env_graph_state = jax.pmap(vstep,
                                                                               out_axes=(0, 0, 0, 0, 0, 0),
                                                                               axis_name="device_axis")(key_step,
                                                                                                        env_state,
                                                                                                        env_act,
                                                                                                        env_graph_state)

                info = jax.tree_map(lambda x: jnp.swapaxes(x.reshape(config["NUM_ENVS"], env.num_agents), 0, 1), info)
                done_batch = batchify(done, env.agents, env.num_agents, config["NUM_ENVS"]).squeeze(axis=2)
                transition = Transition(
                    jnp.full((env.num_agents, config["NUM_ENVS"]), done["__all__"].reshape((config["NUM_ENVS"]))),
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
            key = update_state[-1]

            # metric handling
            metric = jax.tree_map(lambda x: jnp.swapaxes(x, 1, 2), trajectory_batch.info)

            def callback(metric, train_state):
                metric_dict = {
                    "win_rate_global": metric["returned_won_episode"][:, :, 0][metric["returned_episode"][:, :, 0]].mean(),
                    "env_step": metric["update_steps"] * config["NUM_ENVS"] * config["NUM_STEPS"] + env_step_count_init
                }

                for agent in env.agents:
                    metric_dict[f"returns_{agent}"] = metric["returned_episode_returns"][:, :, env.agent_ids[agent]][
                        metric["returned_episode"][:, :, env.agent_ids[agent]]].mean()

                wandb.log(metric_dict)

            metric["update_steps"] = update_steps
            jax.experimental.io_callback(callback, None, metric, train_state)

            update_steps = update_steps + 1

            return ((train_state, env_state, last_obs, last_done, hstate, graph_state, key), update_steps), metric

        runner_state, metric = jax.lax.scan(_run_update, (runner_state, 0), None, config["NUM_UPDATES"])

        return {"runner_state": runner_state, "metrics": metric}

    return train

def run_eval(config, orbax_checkpointer, chkpt_save_path, num_envs=1):
    config["NUM_ENVS"] = num_envs

    if config["DEFINED_PARAM_START"]:
        config["NUM_ENVS"] = config["RESOLUTION"] ** 2

    env = AYS_Environment(reward_type=config["REWARD_TYPE"],
                          num_agents=config["NUM_AGENTS"],
                          homogeneous=config["HOMOGENEOUS"],
                          defined_param_start=config["DEFINED_PARAM_START"],
                          climate_damages=config["CLIMATE_DAMAGES"],
                          evaluating=True)
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"])
    config["MINIBATCH_SIZE"] = (config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"])
    config["CLIP_EPS"] = (config["CLIP_EPS"] / env.num_agents if config["SCALE_CLIP_EPS"] else config["CLIP_EPS"])

    def eval():
        key = jrandom.PRNGKey(config["SEED"])

        if config["DEFINED_PARAM_START"]:
            config["NUM_EVAL_STEPS"] = 2 ** 15
            meshes = jnp.array([jnp.linspace(config["LOWER_BOUND"], config["UPPER_BOUND"], config["RESOLUTION"])] * env.num_agents)
            combined = jnp.array(jnp.meshgrid(*meshes)).T.reshape(-1, env.num_agents)
            default_matrix = jnp.full((combined.shape[0], env.num_agents, env.observation_space(env.agents[0]).shape[0]-1), config["AYS_DEFAULT"])
            initial_states = jnp.insert(default_matrix, config["MID_INDEX"], combined, axis=2).reshape(
                config["NUM_DEVICES"],
                config["NUM_ENVS"] // config["NUM_DEVICES"],
                env.num_agents,
                env.observation_space(env.agents[0]).shape[0])
        else:
            initial_states = jnp.zeros((config["NUM_DEVICES"],
                                        config["NUM_ENVS"] // config["NUM_DEVICES"],
                                        env.num_agents,
                                        env.observation_space(env.agents[0]).shape[0]))

        if env.num_agents == 1:
            actor = Agent(env=env, config=config, key=key)
        else:
            actor = MultiAgent(env=env, config=config, key=key)
        train_state, hstate = actor.initialise()

        target = {'model': train_state}  # must match the input dict
        train_state = orbax_checkpointer.restore(chkpt_save_path, item=target)["model"]

        reset_key = jrandom.split(key, config["NUM_ENVS"]).reshape(config["NUM_DEVICES"],
                                                                   config["NUM_ENVS"] // config["NUM_DEVICES"], -1)

        vreset = jax.jit(jax.vmap(env.reset, in_axes=(0, 0), out_axes=(0, 0, 0), axis_name="batch_axis"))
        obs, env_state, graph_state = jax.pmap(vreset, out_axes=(0, 0, 0), axis_name="device_axis")(reset_key, initial_states)

        runner_state = (
            train_state, env_state, obs, jnp.zeros((env.num_agents, config["NUM_ENVS"]), dtype=bool), hstate,
            graph_state, initial_states,
            key)

        def _eval_step(runner_state, unused):
            # take initial env_state
            train_state, env_state, last_obs, last_done, hstate, env_graph_state, initial_states, key = runner_state

            # act on this initial env_state
            obs_batch = batchify(last_obs, env.agents, env.num_agents, config["NUM_ENVS"])
            hstate, action_n, log_prob_n, value_n, key, pi, spec_key = actor.act(train_state, hstate, obs_batch,
                                                                                 last_done, key)
            env_act = unbatchify(action_n, env.agents, env.num_agents, config["NUM_DEVICES"])
            env_act = {k: v for k, v in env_act.items()}

            # step in env
            key, _key = jrandom.split(key)
            key_step = jrandom.split(key, config["NUM_ENVS"]).reshape(config["NUM_DEVICES"],
                                                                      config["NUM_ENVS"] // config["NUM_DEVICES"],
                                                                      -1)
            vstep = jax.vmap(env.step, in_axes=(0, 0, 0, 0, 0), axis_name="batch_axis")
            obs, env_state, reward, done, info, env_graph_state = jax.pmap(vstep,
                                                                           out_axes=(0, 0, 0, 0, 0, 0),
                                                                           axis_name="device_axis")(key_step,
                                                                                                    env_state,
                                                                                                    env_act,
                                                                                                    env_graph_state,
                                                                                                    initial_states)

            info = jax.tree_map(lambda x: jnp.swapaxes(x.reshape(config["NUM_ENVS"], env.num_agents), 0, 1), info)
            done_batch = batchify(done, env.agents, env.num_agents, config["NUM_ENVS"]).squeeze(axis=2)
            transition = EvalTransition(
                jnp.full((env.num_agents, config["NUM_ENVS"]), done["__all__"].reshape((config["NUM_ENVS"]))),
                done_batch,
                action_n,
                value_n,
                batchify(reward, env.agents, env.num_agents, config["NUM_ENVS"]).squeeze(axis=2),
                log_prob_n,
                obs_batch,
                info,
                pi,
                spec_key,
                env_state,
            )

            # env.render(env_graph_state[0])  # add conditional about rendering or not as depends on jit

            return (train_state, env_state, obs, done_batch, hstate, env_graph_state, initial_states, key), transition

        # run for NUM_STEPS length rollout
        runner_state, trajectory_batch = jax.lax.scan(_eval_step, runner_state, None, config["NUM_EVAL_STEPS"])

        # metric handling
        metric = jax.tree_map(lambda x: jnp.swapaxes(x, 1, 2), trajectory_batch.info)
        metric["update_steps"] = 1

        def callback(metric):
            metric_dict = {
                "eval_returns": metric["returned_episode_returns"][:, :, 0][metric["returned_episode"][:, :, 0]].mean(),
                "eval_win_rate": metric["returned_won_episode"][:, :, 0][metric["returned_episode"][:, :, 0]].mean(),
                "eval_env_step": metric["update_steps"] * config["NUM_ENVS"] * config["NUM_EVAL_STEPS"]
            }

            for agent in env.agents:
                metric_dict[f"eval_returns_{agent}"] = metric["returned_episode_returns"][:, :, env.agent_ids[agent]][
                    metric["returned_episode"][:, :, env.agent_ids[agent]]].mean()
                metric_dict[f"eval_win_rate_{agent}"] = metric["returned_won_episode"][:, :, env.agent_ids[agent]][
                    metric["returned_episode"][:, :, env.agent_ids[agent]]].mean()

            wandb.log(metric_dict)

        # jax.experimental.io_callback(callback, None, metric)  # do we want to use wandb here?

        # below shape is num_eval_steps, num_agents, num_envs, num_actions
        # now converted to num_eval_steps, num_envs, num_agents, num_actions
        distribution_logits = jnp.swapaxes(trajectory_batch.distribution, 1, 2).reshape(
            (config["NUM_EVAL_STEPS"] * config["NUM_ENVS"], env.num_agents, env.action_space(env.agents[0]).n))

        eval_actions = jnp.swapaxes(trajectory_batch.action, 1, 2).reshape(
            (config["NUM_EVAL_STEPS"] * config["NUM_ENVS"], env.num_agents))

        # below shape is num_eval_steps, num_devices, num_envs // num_devices, num_agents, num_actions
        ayse_state = trajectory_batch.env_state.env_state.ayse.reshape(
            (config["NUM_EVAL_STEPS"] * config["NUM_ENVS"], env.num_agents, env.action_space(env.agents[0]).n))

        if config["DEFINED_PARAM_START"]:
            # count how many episodes ended
            eps_ended = jnp.sum(metric["returned_episode"][:, :, 0], axis=0)

            # then do a logical and between end and win
            win_number = jnp.sum(jnp.logical_and(jnp.logical_or(metric["returned_won_episode"][:, :, 0], metric["returned_won_episode"][:, :, 1]), metric["returned_episode"][:, :, 0]), axis=0)
            # compare these two values for each env
            win_ratio = win_number / eps_ended

            matrix_vals = initial_states.reshape((config["NUM_ENVS"],
                                                  env.num_agents,
                                                  env.observation_space(env.agents[0]).shape[0]))[:, :, config["MID_INDEX"]]

            def plot_end_state_matrix(results):
                fig = plt.imshow(jnp.flip(results.reshape(config["RESOLUTION"], config["RESOLUTION"]), axis=0),
                                 extent=(config["LOWER_BOUND"], config["UPPER_BOUND"],
                                         config["LOWER_BOUND"], config["UPPER_BOUND"]),
                                 cmap='mako')
                label = ["Y Agent", "S Agent"]
                plt.ylabel(f'{label[config["MID_INDEX"] - 1]} 0')
                plt.xlabel(f'{label[config["MID_INDEX"] - 1]} 1')
                cbar = plt.colorbar(fig)
                cbar.set_label('Rate of Success')

                ax = plt.gca()

                ax.set_yticklabels([53, 55, 62, 70, 77, 85, ])
                ax.set_xticklabels([53, 55, 62, 70, 77, 85, ])
                ax.set_title('PC')
                plt.tight_layout()
                plt.show()

            def plot_matrix_two(coords, results):
                unique_x = np.unique(coords[:, 0])
                unique_y = np.unique(coords[:, 1])

                color_map = np.zeros((len(unique_y), len(unique_x)))

                for i in range(len(coords)):
                    x, y = coords[i]
                    x_idx = np.where(unique_x == x)[0][0]
                    y_idx = np.where(unique_y == y)[0][0]
                    color_map[y_idx, x_idx] = results[i]

                # Create the plot
                plt.imshow(color_map, extent=(unique_x.min(), unique_x.max(), unique_y.min(), unique_y.max()), cmap="mako")
                plt.show()

            plot_end_state_matrix(win_ratio)
            # plot_matrix_two(matrix_vals, win_ratio)

        def plot_q_differences(input_for_cmap, ayse, agent, title_val, softmax=False, episode_cmap=False):


            cmaps = [sns.color_palette("rocket", as_cmap=True),
                     sns.cubehelix_palette(start=2, rot=0, dark=0, light=0.9, reverse=True, as_cmap=True),
                     sns.color_palette("mako", as_cmap=True),
                     sns.color_palette("mako", as_cmap=True),
                     sns.color_palette("mako", as_cmap=True),
                     sns.color_palette("mako", as_cmap=True),
                     sns.color_palette("mako", as_cmap=True),
                     sns.color_palette("mako", as_cmap=True),
                     sns.color_palette("mako", as_cmap=True),
                     sns.color_palette("mako", as_cmap=True),
                     ]

            fig, ax3d = create_figure_ays(top_down=False)
            a_ind = env.agent_ids[agent]
            colour_diff = input_for_cmap[:, a_ind]
            if softmax:
                # colour_diff = jax.nn.softmax(colour_diff)
                min_value = 0
                max_value = jnp.max(colour_diff)
                colour_diff = (colour_diff - min_value) / (max_value - min_value)

            episode_cmap = False
            # when ayse[:, :, 2] == [0.5] * num_agents then this marks a new episode so should change colour
            if episode_cmap:
                cmaps = [sns.color_palette("mako", as_cmap=True)] * 10
                traj_cmap = jnp.zeros_like(colour_diff)
                increment_val = 0

                for i in range(ayse[:, :, 2].shape[0]):
                    if jnp.array_equal(ayse[i, :, 2], jnp.array([0.5] * env.num_agents)):
                        increment_val += 1
                    traj_cmap = traj_cmap.at[i].set(increment_val)

                traj_cmap /= increment_val
                colour_diff = traj_cmap

            scatter = ax3d.scatter(xs=ayse[:, a_ind, 3], ys=ayse[:, a_ind, 1], zs=ayse[:, a_ind, 0],
                                   c=colour_diff, alpha=0.8, s=1, cmap=cmaps[a_ind])
            legend1 = ax3d.legend(*scatter.legend_elements(),
                                  loc="upper left", title=title_val)
            ax3d.set_title(f"Agent {a_ind}, Reward Func: {env.reward_type[a_ind]}")
            ax3d.add_artist(legend1)

            return fig

        q_max = jnp.max(distribution_logits, axis=2)
        q_avg = jnp.average(distribution_logits, axis=2)
        q_diff = jnp.abs(q_max - q_avg)

        min_value = 0
        max_value = jnp.max(q_diff)
        q_diff = (q_diff - min_value) / (max_value - min_value)

        fig_actions = [plot_q_differences(eval_actions, ayse_state, agent, "Actions") for agent in env.agents]
        fig_q_diff = [plot_q_differences(q_diff, ayse_state, agent, "Logit Diff", softmax=True) for agent in env.agents]

        # return {"runner_state": runner_state, "metrics": metric}
        return fig_actions, fig_q_diff

    return eval



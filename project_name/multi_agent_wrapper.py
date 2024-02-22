import jax
import jax.numpy as jnp
import jax.random as jrandom
from .agents.agent_main import Agent


class MultiAgentWrapper(Agent):
    def __init__(self):
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


        pass

    def act(self):
        pass

    def update(self):
        pass

        # marl stuff
        for agent in env.agents:
            ac_in = (obs_batch[jnp.newaxis, env.agent_ids[agent], :],
                     last_done[jnp.newaxis, env.agent_ids[agent]],
                     )
            ind_hstate, ind_action, ind_log_prob, ind_value, key = agent_list[agent].act(train_state_list[agent],
                                                                                         hstate[env.agent_ids[agent],
                                                                                         :],
                                                                                         ac_in, key)
            action_n = action_n.at[env.agent_ids[agent]].set(ind_action[0])
            value_n = value_n.at[env.agent_ids[agent]].set(ind_value[0])
            log_prob_n = log_prob_n.at[env.agent_ids[agent]].set(ind_log_prob[0])
            hstate = hstate.at[env.agent_ids[agent], :].set(ind_hstate[0])

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

        for agent in env.agents:  # TODO this is probs mega slowsies
            last_obs_batch = batchify(last_obs, env.agents)
            individual_trajectory_batch = jax.tree_map(lambda x: x[:, env.agent_ids[agent]], trajectory_batch)
            individual_train_state = (
                train_state_list[agent], env_state, last_obs_batch[env.agent_ids[agent], :],
                last_done[env.agent_ids[agent]],
                hstate[env.agent_ids[agent], :], key)
            individual_runner_list = agent_list[agent].update(individual_train_state, individual_trajectory_batch)
            train_state_list[agent] = individual_runner_list[0]
            hstate = hstate.at[env.agent_ids[agent], :].set(individual_runner_list[4])
            key = individual_runner_list[-1]
        runner_state = (train_state_list, env_state, last_obs, last_done, hstate, graph_state, key)
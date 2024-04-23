from .AYS_JAX import AYS_Environment
import jax
import jax.random as jrandom


num_agents = 5
key = jax.random.PRNGKey(0)

env = AYS_Environment(reward_type=["PB", "PB", "PB"])

obs, state, graph_states = env.reset(key)

for step in range(200):
    key, key_reset, key_act, key_step = jrandom.split(key, 4)

    # fig = env.render(graph_states)
    # plt.savefig(f"project_name/images/{step}.png")
    # plt.close()
    # print("obs:", obs)

    # Sample random actions.
    key_act = jrandom.split(key_act, env.num_agents)
    actions = {agent: env.action_space(agent).sample(key_act[i]) for i, agent in enumerate(env.agents)}
    actions = {agent: 0 for i, agent in enumerate(env.agents)}

    # print("action:", env.game_actions_idx[actions[env.agents[state.agent_in_room]].item()])

    # Perform the step transition.
    obs, state, reward, done, infos, graph_states = env.step(key_step, state, actions, graph_states)
    # print(state)
    #
    # print("reward:", reward["agent_0"])

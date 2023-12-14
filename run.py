import argparse
from MARL_agent import MARL_agent
import sys
import numpy as np
from datetime import datetime

def main(args):
    if args.chkpt_load:
        chkpt_load_path = args.chkpt_load_path
    else:
        chkpt_load_path = None
    marl_agent = MARL_agent(num_agents=args.num_agents, animation=args.animation, wandb_save=args.wandb_save,
                            model=args.model, test_actions=args.test_actions, top_down=args.top_down,
                            chkpt_load=args.chkpt_load, chkpt_load_path=chkpt_load_path,
                            reward_type=args.reward_type, obs_type=args.observation_type,
                            rational=args.rationality, trade_actions=args.trade_actions, maddpg=args.maddpg,
                            homogeneous=args.homogeneous, seed=args.random_seed, rational_choice=args.rational_choice)

    marl_agent.pretrained_agents_load(maddpg=args.maddpg)

    marl_agent.training_run()
    # marl_agent.env.test_reward_functions()

    # marl_agent.plot_trajectory('red')

    # marl_agent.test_agent()

    # print("Start Time : {}".format(datetime.now()))
    #
    # chkpt_list = ["./checkpoints/env=ays_reward_type=['PB', 'PB']_obs_type=agent_only_num_agents=2_homogeneous=False_rationality=[True, True]_rational_choice=2nd_best_episodes=2000/seed_42.tar",
    #               "./checkpoints/env=ays_reward_type=['PB', 'PB']_obs_type=agent_only_num_agents=2_homogeneous=False_rationality=[True, True]_rational_choice=2nd_best_episodes=2000/seed_15.tar",
    #               "./checkpoints/env=ays_reward_type=['PB', 'PB']_obs_type=agent_only_num_agents=2_homogeneous=False_rationality=[True, True]_rational_choice=2nd_best_episodes=2000/seed_98.tar",
    #               "./checkpoints/env=ays_reward_type=['PB', 'PB']_obs_type=agent_only_num_agents=2_homogeneous=False_rationality=[True, True]_rational_choice=2nd_best_episodes=2000/seed_44.tar",
    #               "./checkpoints/env=ays_reward_type=['PB', 'PB']_obs_type=agent_only_num_agents=2_homogeneous=False_rationality=[True, True]_rational_choice=2nd_best_episodes=2000/seed_22.tar",
    #               "./checkpoints/env=ays_reward_type=['PB', 'PB']_obs_type=agent_only_num_agents=2_homogeneous=False_rationality=[True, True]_rational_choice=2nd_best_episodes=2000/seed_68.tar",
    #               ]
    # seed_list = [42, 15, 98, 44, 22, 68]
    # agent_range = 6
    # results = [0] * agent_range
    # for agent in range(agent_range):
    #     marl_agent = MARL_agent(num_agents=args.num_agents, animation=args.animation, wandb_save=args.wandb_save,
    #                             model=args.model, test_actions=args.test_actions, top_down=args.top_down,
    #                             chkpt_load=args.chkpt_load, chkpt_load_path=chkpt_list[agent],
    #                             reward_type=args.reward_type, obs_type=args.observation_type,
    #                             rational=args.rationality, trade_actions=args.trade_actions, maddpg=args.maddpg,
    #                             homogeneous=args.homogeneous, seed=seed_list[agent],
    #                             rational_choice=args.rational_choice)
    #     marl_agent.pretrained_agents_load(maddpg=args.maddpg)
    #     _, results[agent], upper_bound, lower_bound = marl_agent.test_agent()
    #     print("COMPLETED {} out of {} agents.".format(agent + 1, agent_range))
    #
    # resultant_list = np.array([sum(col) / len(col) for col in zip(*results)])
    # marl_agent.plot_end_state_matrix(resultant_list, upper_bound, lower_bound)
    # print("End Time : {}".format(datetime.now()))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--animation', default=False)
    # parser.add_argument('--animation', default=True)
    parser.add_argument('--top_down', default=False)
    # parser.add_argument('--top_down', default=True)

    parser.add_argument('--wandb_save', default=False)
    # parser.add_argument('--wandb_save', default=True)

    parser.add_argument('--maddpg', default=False)
    # parser.add_argument('--maddpg', default=True)

    parser.add_argument('--homogeneous', default=False)
    # parser.add_argument('--homogeneous', default=True)

    parser.add_argument('--chkpt_load', default=False)
    # parser.add_argument('--chkpt_load', default=True)
    # parser.add_argument('--chkpt_load_path', default="./checkpoints/env=ays_reward_type=['PB', 'PB']_obs_type=agent_only_num_agents=2_episodes=2000/end_time=13-09-2023_10-55-06.tar")  # homo
    # parser.add_argument('--chkpt_load_path', default="./checkpoints/env=ays_reward_type=['PB', 'PB']_obs_type=agent_only_num_agents=2_episodes=2000/end_time=13-09-2023_23-49-53.tar")  # hetero
    # parser.add_argument('--chkpt_load_path', default="./checkpoints/env=ays_reward_type=['PB', 'PB']_obs_type=agent_only_num_agents=2_homogeneous=False_rationality=[True, False]_rational_choice=2nd_best_episodes=2000/end_time=16-09-2023_21-12-05.tar")  # 2nd best irrationality
    # parser.add_argument('--chkpt_load_path', default="./checkpoints/env=ays_reward_type=['PB', 'PB']_obs_type=agent_only_num_agents=2_homogeneous=False_rationality=[True, False]_rational_choice=random_episodes=2000/end_time=15-09-2023_21-04-05.tar")  # random irrationality agent only
    # parser.add_argument('--chkpt_load_path', default="./checkpoints/env=ays_reward_type=['PB', 'PB']_obs_type=all_shared_num_agents=2_homogeneous=False_rationality=[True, False]_rational_choice=random_episodes=2000/end_time=15-09-2023_19-19-57.tar")  # random irrationality all shared

    # parser.add_argument('--reward_type', default=["PB"])
    # parser.add_argument('--reward_type', default=["IPB"])
    # parser.add_argument('--reward_type', default=["max_A"])
    # parser.add_argument('--reward_type', default=["max_Y"])
    parser.add_argument('--reward_type', default=["PB", "PB"])
    # parser.add_argument('--reward_type', default=["PB", "max_Y"])
    # parser.add_argument('--reward_type', default=["IPB", "max_Y"])
    # parser.add_argument('--reward_type', default=["IPB", "max_A"])
    # parser.add_argument('--reward_type', default=["PB", "max_E"])

    parser.add_argument('--observation_type', default="agent_only")
    # parser.add_argument('--observation_type', default="all_shared")

    parser.add_argument('--rationality', default=[True, True])
    # parser.add_argument('--rationality', default=[True, False])
    # parser.add_argument('--rationality', default=[True])
    # parser.add_argument('--rationality', default=[False])

    parser.add_argument('--rational_choice', default="2nd_best")
    # parser.add_argument('--rational_choice', default="random")

    # parser.add_argument('--trade_actions', default=False)  # currently only works for two agents atm
    parser.add_argument('--trade_actions', default=True)

    parser.add_argument('--test_actions', default=False)
    # parser.add_argument('--test_actions', default=True)
    parser.add_argument('--model', type=str, default="ays")
    # parser.add_argument('--model', type=str, default="rice-n")
    parser.add_argument('--num_agents', type=int, default=2)

    parser.add_argument('--random_seed', type=int, default=42)
    # parser.add_argument('--random_seed', type=int, default=15)
    # parser.add_argument('--random_seed', type=int, default=98)
    # parser.add_argument('--random_seed', type=int, default=44)
    # parser.add_argument('--random_seed', type=int, default=22)
    # parser.add_argument('--random_seed', type=int, default=68)

    arguments = parser.parse_args()
    main(arguments)



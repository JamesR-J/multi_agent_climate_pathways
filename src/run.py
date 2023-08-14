import argparse
from MARL_agent import MARL_agent


def main(args):
    marl_agent = MARL_agent(num_agents=args.num_agents, animation=args.animation, wandb_save=args.wandb_save,
                            model=args.model, test_actions=args.test_actions, top_down=args.top_down,
                            chkpt_load=args.chkpt_load, chkpt_load_path=args.chkpt_load_path,
                            reward_type=args.reward_type, obs_type=args.observation_type, load_multi=args.load_multi)

    marl_agent.training_run()
    # marl_agent.env.test_reward_functions()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--animation', default=False)
    parser.add_argument('--animation', default=True)
    parser.add_argument('--top_down', default=False)
    # parser.add_argument('--top_down', default=True)
    parser.add_argument('--wandb_save', default=False)
    # parser.add_argument('--wandb_save', default=True)
    parser.add_argument('--chkpt_load', default=False)
    # parser.add_argument('--chkpt_load', default=True)
    parser.add_argument('--chkpt_load_path', default="./checkpoints/env=ays_reward_type=['PB']_obs_type=agent_only_num_agents=1_episodes=5000/end_time=02-08-2023_19-08-49.tar")
    parser.add_argument('--load_multi', default=False)
    # parser.add_argument('--load_multi', default=True)

    parser.add_argument('--reward_type', default="PB")
    # parser.add_argument('--reward_type', default="posi_negi_PB")
    # parser.add_argument('--reward_type', default="cap_PB")
    # parser.add_argument('--reward_type', default="carbon_reduc")
    # parser.add_argument('--reward_type', default="emission_reduc")

    parser.add_argument('--observation_type', default="agent_only")
    # parser.add_argument('--observation_type', default="all_shared")

    parser.add_argument('--test_actions', default=False)
    # parser.add_argument('--test_actions', default=True)
    parser.add_argument('--model', type=str, default="ays")
    # parser.add_argument('--model', type=str, default="rice-n")
    parser.add_argument('--num_agents', type=int, default=2)

    arguments = parser.parse_args()
    main(arguments)



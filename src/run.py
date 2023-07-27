import argparse
from MARL_agent import MARL_agent


def main(args):
    marl_agent = MARL_agent(num_agents=args.num_agents, animation=args.animation, wandb_save=args.wandb_save,
                            model=args.model, test_actions=args.test_actions, top_down=args.top_down)

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
    parser.add_argument('--test_actions', default=False)
    # parser.add_argument('--test_actions', default=True)
    # parser.add_argument('--model', type=str, default="ays")
    parser.add_argument('--model', type=str, default="rice-n")
    parser.add_argument('--num_agents', type=int, default=2)

    arguments = parser.parse_args()
    main(arguments)



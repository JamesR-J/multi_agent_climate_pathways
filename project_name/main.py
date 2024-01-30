from absl import app, flags
from .MARL_agent import MARL_agent
import sys


_ANIMATION = flags.DEFINE_boolean("animation", False, "save gif of training or not")
_TOP_DOWN = flags.DEFINE_boolean("top_down", False, "top down view or not")
_WANDB = flags.DEFINE_boolean("wandb", True, "wandb or not")

# _RL_ALGO = flags.DEFINE_string("rl_algo", "PPO", "which rl algorithm to use")
_RL_ALGO = flags.DEFINE_string("rl_algo", "D3QN", "which rl algorithm to use")
# parser.add_argument('--algorithm', default="DQN")
# parser.add_argument('--algorithm', default="MADDPG")

_HOMOGENEOUS = flags.DEFINE_boolean("homogeneous", False, "whether to be homogeneous or not")
_CHKPT_LOAD = flags.DEFINE_boolean("chpt_load", False, "whether to load from checkpoint")
# parser.add_argument('--chkpt_load_path', default="./checkpoints/env=ays_reward_type=['PB', 'PB']_obs_type=agent_only_num_agents=2_episodes=2000/end_time=13-09-2023_10-55-06.tar")  # homo

_REWARD_TYPE = flags.DEFINE_list("reward_type", ["PB", "PB"], "what reward function to use")
# parser.add_argument('--reward_type', default=["PB"])
# parser.add_argument('--reward_type', default=["IPB"])
# parser.add_argument('--reward_type', default=["max_A"])
# parser.add_argument('--reward_type', default=["max_Y"])
# parser.add_argument('--reward_type', default=["PB", "PB"])
# parser.add_argument('--reward_type', default=["PB", "max_Y"])
# parser.add_argument('--reward_type', default=["IPB", "max_Y"])
# parser.add_argument('--reward_type', default=["IPB", "max_A"])
# parser.add_argument('--reward_type', default=["PB", "max_E"])

_OBS_TYPE = flags.DEFINE_string("obs_type", "agent_only", "which observation type to use")
# parser.add_argument('--observation_type', default="all_shared")

_RATIONALITY = flags.DEFINE_list("rationality", [True, True], "whether to be rational or not per agent")

_RATIONAL_CHOICE = flags.DEFINE_string("rational_choice", "2nd_best", "which rational choice to use")
# parser.add_argument('--rational_choice', default="random")

_TRADE_ACTIONS = flags.DEFINE_boolean("trade_actions", False, "whether to trade or not, only works for two agents")

_TEST_ACTIONS = flags.DEFINE_boolean("test_actions", False, "whether to use random actions")
_MODEL = flags.DEFINE_string("model", "ays", "ays or rice-n")
_NUM_AGENTS = flags.DEFINE_integer("num_agents", 2, "number of agents")

_SEED = flags.DEFINE_integer("seed", 42, "Random seed")


def main(_):
    if _CHKPT_LOAD.value:
        chkpt_load_path = "need to define this"
    else:
        chkpt_load_path = None
    marl_agent = MARL_agent(num_agents=_NUM_AGENTS.value, animation=_ANIMATION.value, wandb_save=_WANDB.value,
                            model=_MODEL.value, test_actions=_TEST_ACTIONS.value, top_down=_TOP_DOWN.value,
                            chkpt_load=_CHKPT_LOAD.value, chkpt_load_path=chkpt_load_path,
                            reward_type=_REWARD_TYPE.value, obs_type=_OBS_TYPE.value,
                            rational=_RATIONALITY.value, trade_actions=_TRADE_ACTIONS.value, algorithm=_RL_ALGO.value,
                            homogeneous=_HOMOGENEOUS.value, seed=_SEED.value, rational_choice=_RATIONAL_CHOICE.value)

    marl_agent.pretrained_agents_load(algorithm=_RL_ALGO.value)

    marl_agent.training_run()
    # main_main()


if __name__ == '__main__':
    app.run(main)

import sys
import itertools
from ml_collections import config_dict

def get_config():
    config = config_dict.ConfigDict()
    config.LR = 2.5e-4
    config.NUM_ENVS = 128
    config.NUM_STEPS = 256
    config.TOTAL_TIMESTEPS = 100
    config.UPDATE_EPOCHS = 4
    config.NUM_MINIBATCHES = 4
    config.GAMMA = 0.99
    config.GAE_LAMBDA = 0.95
    config.CLIP_EPS = 0.2
    config.ENT_COEF = 0.5
    config.VF_COEF = 0.5
    config.MAX_GRAD_NORM = 0.5
    config.ACTIVATION = "tanh"
    config.ANNEAL_LR = True
    config.GRU_HIDDEN_DIM = 256
    config.SCALE_CLIP_EPS = False

    config.NUM_AGENTS = 2
    config.REWARD_TYPE = ["PB"]
    config.AGENT_TYPE = ["PPO"]

    config.HOMOGENEOUS = False

    config.RUN_TRAIN = True
    config.RUN_EVAL = False
    config.NUM_EVAL_STEPS = 2000

    return config  # TODO get this to work at some point


def sweep_SWEEP():
    # seed_list = [28, 10, 98, 44, 22, 68]
    seed_list = [98]  # 44]
    # homogeneous = [False, True]
    homogeneous = [False]
    # num_agents = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    num_agents = [3]
    # reward_function = ['"PB", "PB", "PB", "PB", "PB", "PB", "PB", "PB", "PB", "PB"']
    # reward_function = ['"PB", "PB"']
    reward_function = ['"PB", "max_A", "max_A"']
    # reward_function = ['"PB", "PB", "PB"', '"PB", "PB", "max_Y"',
    #                    '"PB", "max_A", "max_A"', '"PB", "PB", "max_A"',
    #                    '"PB", "max_Y", "max_A"', '"PB", "max_Y", "max_Y"'
    #                    ]  # TODO assertion for length of this and num agents innit
    num_loops = [5]

    combinations = itertools.product(seed_list, homogeneous, num_agents, reward_function, num_loops)
    result = [{"seed": seed,
               "homogeneous": homo,
               "num_agents": agent,
               "reward_type": reward,
               "split_train": False,
               "num_loops": num_loops,
               "disable_jit": False} for seed, homo, agent, reward, num_loops in combinations]

    return result

# def get_sweep():
#     """Returns a sweep configuration for hyperparameter tuning."""
#
#     config = config_dict.ConfigDict()
#     config.learning_rate = 0.001
#     sweep_config.params["batch_size"] = config_dict.randint(16, 64)
#     # Add other hyperparameters with sweep ranges
#     return sweep

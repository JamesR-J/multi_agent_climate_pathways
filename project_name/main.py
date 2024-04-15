import os
import pickle

from absl import app, flags, logging
import sys
from .envs.AYS_JAX import AYS_Environment, example
import jax
import jax.random as jrandom
import yaml
import wandb
import orbax
import orbax.checkpoint
from flax.training import orbax_utils
from .jaxmarl_ippo import make_train
from .eval_episodes import make_eval
from . import environment_loop
from ml_collections import config_flags
import jax.numpy as jnp
import shutil
import tensorflow as tf
from matplotlib.transforms import Bbox
from .envs.graph_functions import create_figure_ays
from flax.training.train_state import TrainState


_DISABLE_JIT = flags.DEFINE_boolean("disable_jit", False, "jit or not for debugging")
# _DISABLE_JIT = flags.DEFINE_boolean("disable_jit", True, "jit or not for debugging")

_CHKPT_LOAD_PATH = flags.DEFINE_string("chkpt_load_path", None, "whether to load from checkpoint path")
# _CHKPT_LOAD_PATH = flags.DEFINE_string("chkpt_load_path", "single_save_coop_agent_sweep_11", "whether to load from checkpoint path")
# _CHKPT_LOAD_PATH = flags.DEFINE_string("chkpt_load_path", "num_agents=7/coop_sweep_1710169124442_39", "whether to load from checkpoint path")
# _CHKPT_LOAD_PATH = flags.DEFINE_string("chkpt_load_path", "comp_sweep/num_agents=3/comp_sweep_1710354370386_33", "whether to load from checkpoint path")

# _CHKPT_LOAD_PATH = flags.DEFINE_string("chkpt_load_path", "comp_sweep/num_agents=2/comp_sweep_1710341371610_12", "whether to load from checkpoint path")  # "PB, PB
# _CHKPT_LOAD_PATH = flags.DEFINE_string("chkpt_load_path", "comp_sweep/num_agents=2/comp_sweep_1710341371610_10", "whether to load from checkpoint path")  # "PB, max_Y
# _CHKPT_LOAD_PATH = flags.DEFINE_string("chkpt_load_path", "comp_sweep/num_agents=2/comp_sweep_1710341371610_14", "whether to load from checkpoint path")  # "PB, max_A

# _CHKPT_LOAD_PATH = flags.DEFINE_string("chkpt_load_path", "comp_sweep/num_agents=3/comp_sweep_1710354370386_1", "whether to load from checkpoint path")  # "PB, max_Y, max_Y
# _CHKPT_LOAD_PATH = flags.DEFINE_string("chkpt_load_path", "comp_sweep/num_agents=3/comp_sweep_1710354370386_3", "whether to load from checkpoint path")  # "PB, max_A, max_A
# _CHKPT_LOAD_PATH = flags.DEFINE_string("chkpt_load_path", "comp_sweep/num_agents=3/comp_sweep_1710354370386_34", "whether to load from checkpoint path")  # "PB, PB, max_A"

# _CHKPT_LOAD_PATH = flags.DEFINE_string("chkpt_load_path", "single_agent_test/num_agents=1/single_agent_test_1711020058093_1/1099/default", "whether to load from checkpoint path")  # single agent crash test

_SEED = flags.DEFINE_integer("seed", 44, "Random seed")

_WORK_DIR = flags.DEFINE_string("workdir", "orbax_checkpoints", "Work unit directory.")

_NUM_AGENTS = flags.DEFINE_integer("num_agents", 1, "number of agents")

_HOMOGENEOUS = flags.DEFINE_boolean("homogeneous", False, "whether homo or hetero")

# _REWARD_TYPE = flags.DEFINE_list("reward_type", ["PB", "PB", "PB", "PB", "PB", "PB", "PB", "PB", "PB", "PB", "PB"], "which reward functions to use")
_REWARD_TYPE = flags.DEFINE_list("reward_type", ["PB"], "which reward functions to use")
# _REWARD_TYPE = flags.DEFINE_list("reward_type", ["PB", "max_Y"], "which reward functions to use")
# _REWARD_TYPE = flags.DEFINE_list("reward_type", ["PB", "PB", "max_A"], "which reward functions to use")

_CONFIG = config_flags.DEFINE_config_file("config", None, "Config file")
# TODO sort this out so have one total config or something, is a little dodge atm

# TODO is there a way to load the config from checkpoint metadata rather than have to hand manually do it - will figure this out for future
# TODO ie save the config file in same place as the checkpoint and metadata and then somehow load the config from there instead

def main(_):
    # logging.info("FLAGS A")
    # logging.info(flags)
    # logging.info("FLAGS B")
    tf.config.experimental.set_visible_devices([], "GPU")

    # TODO convert the agents to train_state.apply_fn()

    # 2441, num_steps, num_envs, num_agents
    # 3  =   5.60 GB
    # 4  =   8.60 GB
    # 5  =  12.30 GB
    # 6  =  16.57 GB   # 17.91  # 27.60
    # 7  =
    # 8  =  26.86 GB
    # 9  = ~31.00 GB
    # 10 =  39.63 GB

    with open("project_name/ippo_ff.yaml", "r") as file:
        config = yaml.safe_load(file)
    config["SEED"] = _SEED.value
    config["NUM_AGENTS"] = _NUM_AGENTS.value  # TODO improve this as pretty manually intesive and bad coding
    config["HOMOGENEOUS"] = _HOMOGENEOUS.value
    config["REWARD_TYPE"] = _REWARD_TYPE.value
    config["NEW_REWARD_TYPE"] = _REWARD_TYPE.value  # TODO added this for some reason as wandb struggling

    config["AGENT_TYPE"] *= config["NUM_AGENTS"]

    wandb.init(config=config)

    chkpt_dir = os.path.abspath("orbax_checkpoints")
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()

    file_name = os.environ["WANDB_NAME"]
    if _CHKPT_LOAD_PATH.value is None:
        chkpt_name = f'{os.environ["WANDB_RUN_GROUP"]}/num_agents={config["NUM_AGENTS"]}/{file_name}'
    else:
        file_name = _CHKPT_LOAD_PATH.value.split('/')[-1]
        chkpt_name = _CHKPT_LOAD_PATH.value
    chkpt_save_path = f"{chkpt_dir}/{chkpt_name}"

    config["NUM_DEVICES"] = len(jax.local_devices())
    logging.info(f"There are {config['NUM_DEVICES']} GPUs")

    if os.environ["LAUNCH_ON_CLUSTER"] == "True":  # it converts boolean to a string
        copy_to_path = os.path.abspath("../../home/jruddjon/lxm3-staging")  # TODO make this applicable to myriad or beaker
    else:
        copy_to_path = os.path.abspath("../../home/jamesrj_desktop/PycharmProjects/multi_agent_climate_pathways")

    # chkpt = {'model': TrainState(step=0, apply_fn=lambda _: None, params={}, tx={}, opt_state={})}  # TODO re add back in this and the two below
    # save_args = orbax_utils.save_args_from_target(chkpt)
    # orbax_checkpointer.save(chkpt_save_path, chkpt, save_args=save_args)
    # options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=1000, create=True)
    # checkpoint_manager = orbax.checkpoint.CheckpointManager(chkpt_save_path, orbax_checkpointer, options)
    checkpoint_manager = None

    if config["RUN_TRAIN"]:
        ### original that can uncomment out
        # with jax.disable_jit(disable=_DISABLE_JIT.value):
        #     train = jax.jit(environment_loop.run_train(config, checkpoint_manager))  # TODO should this be in a vmap key or not, like jaxmarl? what is more efficient
        #     out = jax.block_until_ready(train())
        #
        # chkpt = {'model': out["runner_state"][0][0]}
        # save_args = orbax_utils.save_args_from_target(chkpt)
        # orbax_checkpointer.save(chkpt_save_path, chkpt, save_args=save_args)

        #### new stuff below to look at, copied the run loop twice so it would work for longer?
        config["TOTAL_TIMESTEPS"] = config["TOTAL_TIMESTEPS"] // 2  # TODO can automate this with a config rather than hardcoding it?
        with jax.disable_jit(disable=_DISABLE_JIT.value):
            train = environment_loop.run_train(config, checkpoint_manager)
            out = jax.block_until_ready(train())

        chkpt = {'model': out["runner_state"][0][0]}
        save_args = orbax_utils.save_args_from_target(chkpt)
        orbax_checkpointer.save(chkpt_save_path, chkpt, save_args=save_args)

        with jax.disable_jit(disable=_DISABLE_JIT.value):
            train = environment_loop.run_train(config, checkpoint_manager, orbax_checkpointer, chkpt_save_path)
            out = jax.block_until_ready(train())

        # delete the original checkpoint in the folder
        shutil.rmtree(chkpt_save_path)

        chkpt = {'model': out["runner_state"][0][0]}
        save_args = orbax_utils.save_args_from_target(chkpt)
        orbax_checkpointer.save(chkpt_save_path, chkpt, save_args=save_args)
        shutil.copytree(chkpt_save_path, f"{copy_to_path}/orbax_checkpoints/{chkpt_name}")
        #### end of the new section

    if config["RUN_EVAL"]:
        if not config["RUN_TRAIN"]:
            shutil.copytree(f"{copy_to_path}/orbax_checkpoints/{chkpt_name}", chkpt_save_path)
        with jax.disable_jit(disable=_DISABLE_JIT.value):
            eval = environment_loop.run_eval(config, orbax_checkpointer, chkpt_save_path, num_envs=4)  # config["NUM_ENVS"])
            fig_actions, fig_q_diff = jax.block_until_ready(eval())
            directory = f"{copy_to_path}/figures/{chkpt_name}"
            if not os.path.exists(directory):
                os.makedirs(directory)
            for agent in range(config["NUM_AGENTS"]):
                fig_actions[agent].savefig(
                    f"{directory}/{file_name}_agent-{agent}_actions.png", bbox_inches="tight")
                fig_q_diff[agent].savefig(
                    f"{directory}/{file_name}_agent-{agent}_q-diff.png", bbox_inches='tight')


if __name__ == '__main__':
    app.run(main)

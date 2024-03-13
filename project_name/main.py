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


_DISABLE_JIT = flags.DEFINE_boolean("disable_jit", False, "jit or not for debugging")
# _DISABLE_JIT = flags.DEFINE_boolean("disable_jit", True, "jit or not for debugging")

_CHKPT_LOAD = flags.DEFINE_boolean("chkpt_load", False, "whether to load from checkpoint")

_SEED = flags.DEFINE_integer("seed", 42, "Random seed")

_WORK_DIR = flags.DEFINE_string("workdir", "orbax_checkpoints", "Work unit directory.")

_NUM_AGENTS = flags.DEFINE_integer("num_agents", 2, "number of agents")

_HOMOGENEOUS = flags.DEFINE_boolean("homogeneous", False, "whether homo or hetero")

_REWARD_TYPE = flags.DEFINE_list("reward_type", ["PB", "PB", "PB", "PB", "PB", "PB", "PB", "PB", "PB", "PB", "PB"], "which reward functions to use")

_CONFIG = config_flags.DEFINE_config_file("config", None, "Config file")
# TODO sort this out so have one total config or something, is a little dodge atm


def main(_):
    # logging.info("FLAGS A")
    # logging.info(flags)
    # logging.info("FLAGS B")
    tf.config.experimental.set_visible_devices([], "GPU")

    # 2441, num_steps, num_envs, num_agents
    # 3  =   5.60 GB
    # 4  =   8.60 GB
    # 5  =
    # 6  =  16.57 GB
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

    # config["REWARD_TYPE"] *= config["NUM_AGENTS"]
    config["AGENT_TYPE"] *= config["NUM_AGENTS"]

    wandb.init(config=config)

    chkpt_dir = os.path.abspath("orbax_checkpoints")
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    file_name = os.environ["WANDB_NAME"]
    chkpt_name = f'{os.environ["WANDB_RUN_GROUP"]}/num_agents={config["NUM_AGENTS"]}/{file_name}'
    # chkpt_name = "coop_sweep_1710158031119_2"  # TODO 5 agents innit
    # chkpt_name = "single_save_coop_agent_sweep_1"  # TODO 2 agents inni
    chkpt_save_path = f"{chkpt_dir}/{chkpt_name}"

    config["NUM_DEVICES"] = len(jax.local_devices())
    logging.info(f"There are {config['NUM_DEVICES']} GPUs")

    if os.environ["LAUNCH_ON_CLUSTER"] == "True":  # it converts boolean to a string
        copy_to_path = os.path.abspath("../../home/jruddjon/lxm3-staging")
    else:
        copy_to_path = os.path.abspath("../../home/jamesrj_desktop/PycharmProjects/multi_agent_climate_pathways")

    if config["RUN_TRAIN"]:
        # pass
    # with jax.profiler.trace("/tmp/tensorboard"):
        with jax.disable_jit(disable=_DISABLE_JIT.value):
            train = jax.jit(environment_loop.run_train(config))  # TODO should this be in a vmap key or not, like jaxmarl? what is more efficient
            out = jax.block_until_ready(train())

            # rng = jax.random.PRNGKey(config["SEED"])
            # train = jax.jit(make_train(config))
            # out = jax.block_until_ready(train(rng))

        chkpt = {'model': out["runner_state"][0][0]}
        save_args = orbax_utils.save_args_from_target(chkpt)
        orbax_checkpointer.save(chkpt_save_path, chkpt, save_args=save_args)
        shutil.copytree(chkpt_save_path, f"{copy_to_path}/orbax_checkpoints/{chkpt_name}")
        # TODO ensure the above still works

    if config["RUN_EVAL"]:
        if not config["RUN_TRAIN"]:
            shutil.copytree(f"{copy_to_path}/orbax_checkpoints/{chkpt_name}", chkpt_save_path)
        with jax.disable_jit(disable=_DISABLE_JIT.value):
            eval = environment_loop.run_eval(config, orbax_checkpointer, chkpt_save_path, num_envs=4)  # config["NUM_ENVS"])  # TODO why can't jit this?
            fig_actions, fig_q_diff = jax.block_until_ready(eval())
            directory = f"{copy_to_path}/figures/{chkpt_name}"
            if not os.path.exists(directory):
                os.makedirs(directory)
            fig_actions.savefig(f"{directory}/{file_name}_actions.png")
            fig_q_diff.savefig(f"{directory}/{file_name}_q_diff.png")


if __name__ == '__main__':
    app.run(main)

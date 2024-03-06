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


_WANDB = flags.DEFINE_boolean("wandb", False, "wandb or not")
# _WANDB = flags.DEFINE_boolean("wandb", True, "wandb or not")

_DISABLE_JIT = flags.DEFINE_boolean("disable_jit", False, "jit or not for debugging")
# _DISABLE_JIT = flags.DEFINE_boolean("disable_jit", True, "jit or not for debugging")

_CHKPT_LOAD = flags.DEFINE_boolean("chkpt_load", False, "whether to load from checkpoint")

_SEED = flags.DEFINE_integer("seed", 42, "Random seed")

_WORK_DIR = flags.DEFINE_string("workdir", "checkpoints", "Work unit directory.")

_CONFIG = config_flags.DEFINE_config_file("config", None, "Config file")
# TODO sort this out so have one total config or something, is a little dodge atm


def main(_):
    with open("project_name/ippo_ff.yaml", "r") as file:
        config = yaml.safe_load(file)
    config["SEED"] = _SEED.value

    config["REWARD_TYPE"] *= config["NUM_AGENTS"]
    config["AGENT_TYPE"] *= config["NUM_AGENTS"]

    if _WANDB.value:
        wandb_mode = "online"
    else:
        wandb_mode = "disabled"

    wandb.init(entity="jamesr-j",
               config=config,
               mode=wandb_mode)

    ckpt_dir = '/tmp/flax_chkpt'
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    chkpt_save_path = '/tmp/flax_chkpt/single_save_' + str(config["SEED"])

    config["NUM_DEVICES"] = len(jax.local_devices())
    logging.info(f"There are {config['NUM_DEVICES']} GPUs")

    os.makedirs(_WORK_DIR.value, exist_ok=True)

    # jax.profiler.start_trace("/tmp/tensorboard")

    if config["RUN_TRAIN"]:
        if os.path.exists(ckpt_dir):
            shutil.rmtree(ckpt_dir)

        with jax.disable_jit(disable=_DISABLE_JIT.value):
            train = jax.jit(environment_loop.run_train(config))  # TODO should this be in a vmap key or not, like jaxmarl? what is more efficient
            out = jax.block_until_ready(train())

            # rng = jax.random.PRNGKey(config["SEED"])
            # train = jax.jit(make_train(config))
            # out = jax.block_until_ready(train(rng))

            # jax.profiler.stop_trace()

        chkpt = {'model': out["runner_state"][0][0]}
        save_args = orbax_utils.save_args_from_target(chkpt)
        orbax_checkpointer.save(chkpt_save_path, chkpt, save_args=save_args)

    if config["RUN_EVAL"]:
        # CHECKPOINTING
        # Some arbitrary nested pytree with a dictionary and a NumPy array.
        # config_chkpt = {'dimensions': np.array([5, 3])}  # TODO understand this

        # save_args = orbax_utils.save_args_from_target(ckpt)
        # orbax_checkpointer.save('./project_name/orbax_saves/single_save', ckpt)#, save_args=save_args)
        with jax.disable_jit(disable=False):
            # eval = jax.jit(environment_loop.run_eval(config, orbax_checkpointer, chkpt_save_path))  # TODO why can't jit this?
            eval = environment_loop.run_eval(config, orbax_checkpointer, chkpt_save_path)
            out = jax.block_until_ready(eval())

    # from .jaxmarl_iql import make_train
    # with open("project_name/iql.yaml", "r") as file:
    #     config = yaml.safe_load(file)
    # config["SEED"] = _SEED.value
    # config["NUM_AGENTS"] = _NUM_AGENTS.value
    # config["REWARD_TYPE"] = _REWARD_TYPE.value * _NUM_AGENTS.value
    #
    # rng = jax.random.PRNGKey(config["SEED"])
    # rngs = jax.random.split(rng, config["NUM_SEEDS"])
    # train_vjit = jax.jit(jax.vmap(make_train(config)))
    # with jax.disable_jit(disable=_DISABLE_JIT.value):
    #     outs = jax.block_until_ready(train_vjit(rngs))


if __name__ == '__main__':
    app.run(main)

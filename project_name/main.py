import os
import pickle

from absl import app, flags
import sys
from .envs.AYS_JAX import AYS_Environment, example
import jax
import jax.random as jrandom
import yaml
import wandb
import orbax
import orbax.checkpoint
from .jaxmarl_ippo import make_train
from .eval_episodes import make_eval
from . import environment_loop
from ml_collections import config_flags


_WANDB = flags.DEFINE_boolean("wandb", False, "wandb or not")
# _WANDB = flags.DEFINE_boolean("wandb", True, "wandb or not")

_DISABLE_JIT = flags.DEFINE_boolean("disable_jit", False, "jit or not for debugging")
# _DISABLE_JIT = flags.DEFINE_boolean("disable_jit", True, "jit or not for debugging")

_RUN_EVAL = flags.DEFINE_boolean("run_eval", False, "run evaluation steps or not")
# _RUN_EVAL = flags.DEFINE_boolean("run_eval", True, "run evaluation steps or not")

_CHKPT_LOAD = flags.DEFINE_boolean("chkpt_load", False, "whether to load from checkpoint")
# parser.add_argument('--chkpt_load_path', default="./checkpoints/env=ays_reward_type=['PB', 'PB']_obs_type=agent_only_num_agents=2_episodes=2000/end_time=13-09-2023_10-55-06.tar")  # homo

_OBS_TYPE = flags.DEFINE_string("obs_type", "agent_only", "which observation type to use")
# parser.add_argument('--observation_type', default="all_shared")

_TEST_ACTIONS = flags.DEFINE_boolean("test_actions", False, "whether to use random actions")

_SEED = flags.DEFINE_integer("seed", 42, "Random seed")

_CONFIG = config_flags.DEFINE_config_file("config", None, "Config file")
# TODO sort this out so have one total config or something, is a little dodge atm


def main(_):
    # print(_SEED.value)
    # print(_WANDB.value)
    # sys.exit()

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

    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    chkpt_save_path = "./lxm3-staging/checkpoints/single_save_" + str(config["SEED"])

    # train_jit = jax.jit(make_train(config, orbax_checkpointer), device=jax.devices()[0])
    with jax.disable_jit(disable=_DISABLE_JIT.value):
        out = environment_loop.run_train(config)  # TODO why can't I wrap this in a jax.jit?
        chkpt = {'model': out["runner_state"][0][0]}
        # chkpt_save_path = '/jruddjon/lxm/flax_ckpt/orbax/single_save_' + str(config["SEED"])
        # orbax_checkpointer.save(chkpt_save_path, ckpt)

        # with open(chkpt_save_path, 'wb') as file:
        #     pickle.dump(chkpt, file)
    if _RUN_EVAL.value:
        # CHECKPOINTING
        # Some arbitrary nested pytree with a dictionary and a NumPy array.
        # config_chkpt = {'dimensions': np.array([5, 3])}  # TODO understand this

        # save_args = orbax_utils.save_args_from_target(ckpt)
        # orbax_checkpointer.save('./project_name/orbax_saves/single_save', ckpt)#, save_args=save_args)
        with jax.disable_jit(disable=True):
            out = environment_loop.run_eval(config, orbax_checkpointer, chkpt_save_path)  # TODO why can't I wrap this in a jax.jit?

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

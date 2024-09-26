import os
from absl import app, flags, logging
import jax
import yaml
import wandb
from . import environment_loop
from ml_collections import config_flags
import orbax
import orbax.checkpoint
from flax.training import orbax_utils
from flax.training.train_state import TrainState
import shutil
from datetime import datetime


_DISABLE_JIT = flags.DEFINE_boolean("disable_jit", False, "jit or not for debugging")
_CHKPT_LOAD_PATH = flags.DEFINE_string("chkpt_load_path", None, "whether to load from checkpoint path")
_SEED = flags.DEFINE_integer("seed", 44, "Random seed")
_WORK_DIR = flags.DEFINE_string("workdir", "orbax_checkpoints", "Work unit directory.")
_NUM_AGENTS = flags.DEFINE_integer("num_agents", 2, "number of agents")
_REWARD_TYPE = flags.DEFINE_list("reward_type", ["PB", "PB"], "Agent reward types")
_HOMOGENEOUS = flags.DEFINE_boolean("homogeneous", False, "whether homo or hetero")
_CLIMATE_DAMAGES = flags.DEFINE_list("climate_damages", ["1", "0.25"], " climate damages stuff")
_SPLIT_TRAIN = flags.DEFINE_boolean("split_train", False, "whether to run looped training or not")
_NUM_LOOPS = flags.DEFINE_integer("num_loops", 2, "number of loops for split train")


def main(_):
    with open("project_name/ippo_config_global.yaml", "r") as file:
        config = yaml.safe_load(file)
    config["SEED"] = _SEED.value
    config["NUM_AGENTS"] = _NUM_AGENTS.value
    config["REWARD_TYPE"] = _REWARD_TYPE.value
    config["HOMOGENEOUS"] = _HOMOGENEOUS.value
    config["SPLIT_TRAIN"] = _SPLIT_TRAIN.value
    config["NUM_LOOPS"] = _NUM_LOOPS.value
    config["CLIMATE_DAMAGES"] = _CLIMATE_DAMAGES.value

    config["AGENT_TYPE"] *= config["NUM_AGENTS"]

    wandb_name = "testing_" + str(datetime.now())
    wandb_run_group = "tests_1"

    wandb.init(config=config,
               project="TEST",
               name=wandb_name,
               group=wandb_run_group,
               mode="disabled")

    chkpt_dir = os.path.abspath("orbax_checkpoints")
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()

    file_name = wandb_name
    if _CHKPT_LOAD_PATH.value is None:
        chkpt_name = f'{wandb_run_group}/num_agents={config["NUM_AGENTS"]}/{file_name}'
    else:
        file_name = _CHKPT_LOAD_PATH.value.split('/')[-1]
        chkpt_name = _CHKPT_LOAD_PATH.value
    chkpt_save_path = f"{chkpt_dir}/{chkpt_name}"

    config["NUM_DEVICES"] = len(jax.local_devices())
    logging.info(f"There are {config['NUM_DEVICES']} GPUs")

    copy_to_path = os.path.abspath("multi_agent_climate_pathways")

    chkpt = {'model': TrainState(step=0, apply_fn=lambda _: None, params={}, tx={}, opt_state={})}
    save_args = orbax_utils.save_args_from_target(chkpt)
    orbax_checkpointer.save(chkpt_save_path, chkpt, save_args=save_args)
    options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=1000, create=True)
    checkpoint_manager = orbax.checkpoint.CheckpointManager(chkpt_save_path, orbax_checkpointer, options)

    if config["RUN_TRAIN"]:
        if config["SPLIT_TRAIN"]:
            config["TOTAL_TIMESTEPS"] = config["TOTAL_TIMESTEPS"] // config["NUM_LOOPS"]
            env_step_count_init = 0
            train_state_input = None
            for loop in range(config["NUM_LOOPS"]):
                with jax.disable_jit(disable=_DISABLE_JIT.value):
                    train = jax.jit(environment_loop.run_train(config,
                                                               checkpoint_manager,
                                                               env_step_count_init=env_step_count_init,
                                                               train_state_input=train_state_input))

                    out = jax.block_until_ready(train())
                    env_step_count_init = (out["runner_state"][1] * (loop + 1)) * config["NUM_ENVS"] * config["NUM_STEPS"]
                    train_state_input = out["runner_state"][0][0]

        else:
            with jax.disable_jit(disable=_DISABLE_JIT.value):
                train = jax.jit(environment_loop.run_train(config, checkpoint_manager))
                out = jax.block_until_ready(train())

        chkpt = {'model': out["runner_state"][0][0]}
        save_args = orbax_utils.save_args_from_target(chkpt)
        orbax_checkpointer.save(chkpt_save_path, chkpt, save_args=save_args)
        shutil.copytree(chkpt_save_path, f"{copy_to_path}/orbax_checkpoints/{chkpt_name}")

    if config["RUN_EVAL"]:
        with jax.disable_jit(disable=_DISABLE_JIT.value):
            eval = environment_loop.run_eval(config, orbax_checkpointer, chkpt_save_path, num_envs=4)
            fig_actions, fig_q_diff = jax.block_until_ready(eval())
            directory = f"{copy_to_path}/figures/{chkpt_name}"
            if not os.path.exists(directory):
                os.makedirs(directory)
            for agent in range(config["NUM_AGENTS"]):
                fig_actions[agent].savefig(
                    f"{directory}/{file_name}_agent-{agent}_actions.svg")
                fig_q_diff[agent].savefig(
                    f"{directory}/{file_name}_agent-{agent}_q-diff.svg")


if __name__ == '__main__':
    app.run(main)

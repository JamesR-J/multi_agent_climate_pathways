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


_WANDB = flags.DEFINE_boolean("wandb", False, "wandb or not")
# _WANDB = flags.DEFINE_boolean("wandb", True, "wandb or not")

_DISABLE_JIT = flags.DEFINE_boolean("disable_jit", False, "jit or not for debugging")
# _DISABLE_JIT = flags.DEFINE_boolean("disable_jit", True, "jit or not for debugging")

_RUN_EVAL = flags.DEFINE_boolean("run_eval", False, "run evaluation steps or not")
# _RUN_EVAL = flags.DEFINE_boolean("run_eval", True, "run evaluation steps or not")

_HOMOGENEOUS = flags.DEFINE_boolean("homogeneous", False, "whether to be homogeneous or not")
_CHKPT_LOAD = flags.DEFINE_boolean("chkpt_load", False, "whether to load from checkpoint")
# parser.add_argument('--chkpt_load_path', default="./checkpoints/env=ays_reward_type=['PB', 'PB']_obs_type=agent_only_num_agents=2_episodes=2000/end_time=13-09-2023_10-55-06.tar")  # homo

_REWARD_TYPE = flags.DEFINE_list("reward_type", ["PB"], "what reward function to use")
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

_TEST_ACTIONS = flags.DEFINE_boolean("test_actions", False, "whether to use random actions")
_NUM_AGENTS = flags.DEFINE_integer("num_agents", 3, "number of agents")

_SEED = flags.DEFINE_integer("seed", 42, "Random seed")


def main(_):
    # if _CHKPT_LOAD.value:
    #     chkpt_load_path = "need to define this"
    # else:
    #     chkpt_load_path = None
    # marl_agent = MARL_agent(num_agents=_NUM_AGENTS.value, animation=_ANIMATION.value, wandb_save=_WANDB.value,
    #                         model=_MODEL.value, test_actions=_TEST_ACTIONS.value, top_down=_TOP_DOWN.value,
    #                         chkpt_load=_CHKPT_LOAD.value, chkpt_load_path=chkpt_load_path,
    #                         reward_type=_REWARD_TYPE.value, obs_type=_OBS_TYPE.value,
    #                         rational=_RATIONALITY.value, trade_actions=_TRADE_ACTIONS.value, algorithm=_RL_ALGO.value,
    #                         homogeneous=_HOMOGENEOUS.value, seed=_SEED.value, rational_choice=_RATIONAL_CHOICE.value)
    #
    # marl_agent.pretrained_agents_load(algorithm=_RL_ALGO.value)
    #
    # marl_agent.training_run()
    # main_main()
    # key = jrandom.PRNGKey(_SEED.value)
    # env = AYS_Environment()
    # with jax.disable_jit():
    #     example()

    with open("project_name/ippo_ff.yaml", "r") as file:
        config = yaml.safe_load(file)
    config["SEED"] = _SEED.value
    config["NUM_AGENTS"] = _NUM_AGENTS.value
    config["REWARD_TYPE"] = _REWARD_TYPE.value * _NUM_AGENTS.value

    if _WANDB.value:
        wandb_mode = "online"
    else:
        wandb_mode = "disabled"

    wandb.init(entity="jamesr-j",
               config=config,
               mode=wandb_mode)

    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()

    rng = jax.random.PRNGKey(config["SEED"])
    # train_jit = jax.jit(make_train(config, orbax_checkpointer), device=jax.devices()[0])
    with jax.disable_jit(disable=_DISABLE_JIT.value):
        out = environment_loop.run_train(config)
        # out = train_jit(rng)

    if _RUN_EVAL.value:
        # CHECKPOINTING
        # Some arbitrary nested pytree with a dictionary and a NumPy array.
        # config_chkpt = {'dimensions': np.array([5, 3])}  # TODO understand this
        ckpt = {'model': out["runner_state"][0][0]}
        # save_args = orbax_utils.save_args_from_target(ckpt)
        # orbax_checkpointer.save('./project_name/orbax_saves/single_save', ckpt)#, save_args=save_args)
        orbax_checkpointer.save('/tmp/flax_ckpt/orbax/single_save', ckpt)

        rng = jax.random.PRNGKey(config["SEED"])
        train_jit = jax.jit(make_eval(config, orbax_checkpointer), device=jax.devices()[0])
        with jax.disable_jit(disable=True):
            out = train_jit(rng)

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

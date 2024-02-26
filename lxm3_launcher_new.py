import functools
import importlib
import importlib.util
import os
import subprocess
import sys
from absl import app
from absl import flags
from absl import logging
from lxm3 import xm
from lxm3 import xm_cluster
from lxm3.contrib import ucl
from ml_collections import config_flags

with open("wandb_api_key.txt", "r") as file:
    wandb_api_key = file.read().strip()

_LAUNCH_ON_CLUSTER = flags.DEFINE_boolean("launch_on_cluster", False, "Launch on cluster")
_SINGULARITY_CONTAINER = flags.DEFINE_string(
    "container", "docker-daemon://finetune:latest", "Path to singularity container"
)
_EXP_NAME = flags.DEFINE_string("exp_name", None, "Name of experiment")
_DRY_RUN = flags.DEFINE_bool("dry_run", None, "Dry run")
_ENTRYPOINT = flags.DEFINE_string("entrypoint", None, "Module of the entrypoint")
# _SWEEP = flags.DEFINE_string("sweep", None, "Name of the sweep")
_SWEEP = flags.DEFINE_string("sweep", "SWEEP", "Name of the sweep")
_SWEEP_INDEX = flags.DEFINE_string("sweep_index", "4", "Index of configuration in the sweep")
_TIMEOUT = flags.DEFINE_integer("timeout", 48, "Timeout in hours")
_WANDB_GROUP = flags.DEFINE_string("wandb_group", "{xid}_{name}", "wandb group")
_WANDB_PROJECT = flags.DEFINE_string("wandb_project", "multi_agent_climate_pathways",
                                     "wandb project")
_WANDB_ENTITY = flags.DEFINE_string("wandb_entity", "jamesr-j", "wandb entity")
_WANDB_MODE = flags.DEFINE_string("wandb_mode", "online", "wandb mode")
config_flags.DEFINE_config_file("config", None, "Path to config")
flags.mark_flags_as_required(["config", "entrypoint"])
FLAGS = flags.FLAGS


@functools.lru_cache()
def _get_vcs_info():
    vcs = None
    try:
        import vcsinfo
        vcs_root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], text=True
        ).strip()
        vcs = vcsinfo.detect_vcs(vcs_root)
    except subprocess.SubprocessError:
        logging.warn("Failed to detect VCS info")
    return vcs


def _get_hyper():
    if _SWEEP.value is not None:
        sweep_file = config_flags.get_config_filename(FLAGS["config"])
        sys.path.insert(0, os.path.abspath(os.path.dirname(sweep_file)))
        sweep_module, _ = os.path.splitext(os.path.basename(sweep_file))
        m = importlib.import_module(sweep_module)
        sys.path.pop(0)
        sweep_fn_name = f"sweep_{_SWEEP.value}"
        logging.info(f"Running sweep {sweep_fn_name}")
        sweep_fn = getattr(m, sweep_fn_name, None)
        if sweep_fn is None:
            raise ValueError(f"Sweep {sweep_fn_name} does not exist in {sweep_file}")
        else:
            return sweep_fn()
    else:
        return [{}]


def main(_):
    exp_name = _EXP_NAME.value
    if exp_name is None:
        exp_name = _ENTRYPOINT.value.replace(".", "_")
    if importlib.util.find_spec(_ENTRYPOINT.value) is None:
        raise app.UsageError(f"Entrypoint {_ENTRYPOINT.value} does not exist")
    with xm_cluster.create_experiment(experiment_title=exp_name) as experiment:
        job_requirements = xm_cluster.JobRequirements(gpu=1, ram=8 * xm.GB)
        env_vars = {"XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
        if _LAUNCH_ON_CLUSTER.value:
            # TODO: Make this configurable for non-UCL clusters
            # tfds_data_dir = "/cluster/project0/offline_rl/tensorflow_datasets"
            # d4rl_dataset_dir = "/cluster/project0/offline_rl/d4rl"
            executor = ucl.UclGridEngine(
                job_requirements,
                walltime=_TIMEOUT.value * xm.Hr,
                extra_directives=["-l gpu_type=gtx1080ti"],  # TODO allows specifying GPU type on a cluster
                singularity_options=xm_cluster.SingularityOptions(
                    # bind={
                    #     tfds_data_dir: tfds_data_dir,
                    #     d4rl_dataset_dir: d4rl_dataset_dir,
                    # }
                ),
            )
            # env_vars["TFDS_DATA_DIR"] = tfds_data_dir
            # env_vars["D4RL_DATASET_DIR"] = d4rl_dataset_dir
        else:
            # tfds_data_dir = "/mnt/data/tensorflow_datasets"
            executor = xm_cluster.Local(
                job_requirements,
                singularity_options=xm_cluster.SingularityOptions(
                    # bind={tfds_data_dir: tfds_data_dir}
                ),
            )
            # env_vars["TFDS_DATA_DIR"] = tfds_data_dir
        config_resource = xm_cluster.Fileset(files={config_flags.get_config_filename(FLAGS["config"]): "lxm3_config.py"})
        spec = xm_cluster.PythonPackage(path=".",
                                        entrypoint=xm_cluster.ModuleName(_ENTRYPOINT.value),
                                        resources=[config_resource],
                                        )
        singularity_container = _SINGULARITY_CONTAINER.value
        if singularity_container:
            spec = xm_cluster.SingularityContainer(spec, image_path=singularity_container)
        args = {"config": config_resource.get_path("lxm3_config.py", executor.Spec())}  # type: ignore
        overrides = config_flags.get_override_values(FLAGS["config"])
        overrides = {f"config.{k}": v for k, v in overrides.items()}
        logging.info("Overrides: %r", overrides)
        args.update(overrides)
        sweep = list(_get_hyper())
        if _SWEEP_INDEX.value is not None:
            filtered_sweep = []
            for index in _SWEEP_INDEX.value.split(","):
                filtered_sweep.append(sweep[int(index)])
            sweep = filtered_sweep
        print(sweep)
        sys.exit()
        logging.info("Will launch %d jobs", len(sweep))
        if _DRY_RUN.value:
            logging.info("Will launch %d jobs with the following parameters", len(sweep))
            for i, parameters in enumerate(sweep):
                print(i, parameters)
            return
        [executable] = experiment.package(
            [
                xm.Packageable(
                    spec, executor_spec=executor.Spec(), args=args, env_vars=env_vars
                )
            ]
        )
        xid = experiment.experiment_id
        experiment_name = exp_name
        vcs = _get_vcs_info()
        commit_envs = {}
        if vcs is not None:
            commit_envs["WANDB_GIT_REMOTE_URL"] = vcs.upstream_repo
            commit_envs["WANDB_GIT_COMMIT"] = vcs.id
        envs = [
            {
                "WANDB_API_KEY": wandb_api_key,
                "WANDB_PROJECT": _WANDB_PROJECT.value,
                "WANDB_ENTITY": _WANDB_ENTITY.value,
                "WANDB_NAME": f"{experiment_name}_{xid}_{wid}",
                "WANDB_MODE": _WANDB_MODE.value,
                "WANDB_RUN_GROUP": _WANDB_GROUP.value.format(
                    name=experiment_name, xid=xid
                ),
                **commit_envs,
            }
            for wid in range(len(sweep))
        ]
        experiment.add(
            xm_cluster.ArrayJob(executable, executor, args=sweep, env_vars=envs)
        )


if __name__ == "__main__":
    app.run(main)

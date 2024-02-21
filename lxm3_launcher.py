#!/usr/bin/env python3
# type: ignore
from absl import app
from absl import flags
from lxm3 import xm
from lxm3 import xm_cluster
from lxm3.contrib import ucl
import sys

with open("wandb_api_key.txt", "r") as file:
    wandb_api_key = file.read().strip()

_LAUNCH_ON_CLUSTER = flags.DEFINE_boolean(
    "launch_on_cluster", False, "Launch on cluster"
)
_USE_GPU = flags.DEFINE_boolean("use_gpu", True, "If set, use GPU")
_SINGULARITY_CONTAINER = flags.DEFINE_string(
    "container", None, "Path to singularity container"
)
_ENTRYPOINT = flags.DEFINE_string("entrypoint", None, "Entrypoint for experiment")
flags.mark_flags_as_required(["entrypoint"])


def main(_):
    with xm_cluster.create_experiment(experiment_title="basic") as experiment:
        if _USE_GPU.value:
            job_requirements = xm_cluster.JobRequirements(gpu=1, ram=8 * xm.GB)
        else:
            job_requirements = xm_cluster.JobRequirements(ram=8 * xm.GB)
        if _LAUNCH_ON_CLUSTER.value:
            # This is a special case for using SGE in UCL where we use generic
            # job requirements and translate to SGE specific requirements.
            # Non-UCL users, use `xm_cluster.GridEngine directly`.
            executor = ucl.UclGridEngine(
                job_requirements,
                walltime=60 * 48 * xm.Min,
                extra_directives=["-l gpu_type=gtx1080ti"]  # TODO allows specifying GPU type on a cluster
            )
        else:
            executor = xm_cluster.Local(job_requirements)

        spec = xm_cluster.PythonPackage(
            # This is a relative path to the launcher that contains
            # your python package (i.e. the directory that contains pyproject.toml)
            path=".",
            # Entrypoint is the python module that you would like to
            # In the implementation, this is translated to
            #   python3 -m py_package.main
            entrypoint=xm_cluster.ModuleName(_ENTRYPOINT.value),
        )

        # Wrap the python_package to be executing in a singularity container.
        singularity_container = _SINGULARITY_CONTAINER.value

        # It's actually not necessary to use a container, without it, we
        # fallback to the current python environment for local executor and
        # whatever Python environment picked up by the cluster for GridEngine.
        # For remote execution, using the host environment is not recommended.
        # as you may spend quite some time figuring out dependency problems than
        # writing a simple Dockfiler/Singularity file.
        if singularity_container is not None:
            spec = xm_cluster.SingularityContainer(
                spec,
                image_path=singularity_container,
            )

        [executable] = experiment.package(
            [xm.Packageable(spec, executor_spec=executor.Spec())]
        )

        """
        SINGLE RUN BELOW
        """
        # experiment.add(
        #     xm.Job(
        #         executable=executable,
        #         executor=executor,
        #         # You can pass additional arguments to your executable with args
        #         # This will be translated to `--seed 1`
        #         # Note for booleans we currently use the absl.flags convention
        #         # so {'gpu': False} will be translated to `--nogpu`
        #         # args={"seed": 1},
        #         # You can customize environment_variables as well.
        #         args={"wandb": True},
        #         env_vars={"XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        #                   "WANDB_API_KEY": wandb_api_key,
        #                   "WANDB_PROJECT": "climate_pathways",
        #                   "WANDB_RUN_GROUP": "intro_tests"}
        #     )
        # )

        """ 
        BATCH RUN BELOW
        """
        # To submit parameter sweep by array jobs, you can use the batch context
        # Without the batch context, jobs will be submitted individually.
        seed_list = [42, 15, 98, 44, 22, 68]
        args = [{"seed": seed} for seed in seed_list]
        batch_name = "test_ppo "
        env_vars = [{"XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                     "WANDB_API_KEY": wandb_api_key,
                     "WANDB_PROJECT": "climate_pathways",
                     "WANDB_RUN_GROUP": batch_name,
                     "WANDB_NAME": f"{batch_name} - seed={seed}",
                     "TASK": f"foo_{seed}"} for seed in seed_list]
        experiment.add(
            xm_cluster.ArrayJob(
                executable=executable, executor=executor, args=args, env_vars=env_vars
            )
        )


if __name__ == "__main__":
    app.run(main)

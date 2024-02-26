import sys


def get_config():
    """
    Has:
    exp_name
    sweep
    sweep_index
    wandb_group
    wandb_project
    """

    seed_list = [42, 15, 98, 44, 22, 68]

    def sweep_func():
        return "True"

    # return {"wandb_project": "multi_agent_climate_pathways",
    #         'exp_name': "Testing tings",
    #         "sweep": "Seed Range",
    #         "wandb": True,
    #         "sweep_index": [{"seed": seed} for seed in seed_list],
    #         }

    return {"sweep_SWEEP": sweep_func(), "timeout": 2354}


def sweep_SWEEP():
    seed_list = [42, 15, 98, 44, 22, 68]
    return {"wandb_project": "multi_agent_climate_pathways",
            'exp_name': "Testing tings",
            "sweep": "Seed Range",
            "wandb": True,
            "seed": [{"seed": seed} for seed in seed_list],
            }

import pandas as pd
import wandb
import sys
import matplotlib.pyplot as plt
import seaborn as sns


with open("wandb_api_key.txt", "r") as file:
    wandb_api_key = file.read().strip()

api = wandb.Api()
entity, project = "jamesr-j", "multi_agent_climate_pathways"
runs = api.runs(entity + "/" + project)

def keep_relevant_data(df, group_col, name_col):
    df = df[['env_step', 'win_rate']].copy()  # Keep only relevant columns
    df['group'] = group_col  # Add group information
    df['name'] = name_col  # Add name information
    return df

run_names = []
for run in runs:
    # # .summary contains the output keys/values
    # #  for metrics such as accuracy.
    # #  We call ._json_dict to omit large files
    # summary_list.append(run.summary._json_dict)
    #
    # # .config contains the hyperparameters.
    # #  We remove special values that start with _.
    # config_list.append({k: v for k, v in run.config.items() if not k.startswith("_")})
    #
    # # .name is the human-readable name of the run.
    # name_list.append(run.name)
    # print(run.group)
    if run.state == "finished":
        run_names.append({"name": run.name, "group": run.group, "id": run.id})
        # dfs.append(pd.DataFrame({"name": run.name, "group": run.group, "data": run.history()}))

print(run_names)
print(pd.DataFrame.from_dict(run_names))

# # Apply the function to each DataFrame in the list and store the result
# # processed_dfs = [keep_relevant_data(df, df['group'], df['name']) for df in dfs]
#
# # Concatenate the processed DataFrames
# result = pd.concat(processed_dfs, ignore_index=True)
#
# # Group the result by 'group' and keep track of 'name' through a list
# grouped_result = result.groupby('group')['name'].agg(list).reset_index()
#
# print(grouped_result)

# runs_df.to_csv("project.csv")

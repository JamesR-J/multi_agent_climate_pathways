import pandas as pd
import matplotlib.pyplot as plt
import sys
import seaborn as sns


sns.set(font_scale=0.8)


df1 = pd.read_csv("./wandb_csv_data/maxY_actions.csv")
df2 = pd.read_csv("./wandb_csv_data/maxY_episode_rewards.csv")

merged_df = pd.merge(df1, df2, on='Step', how='outer').sort_values('Step')

print(df1.info())
print(df2.info())

print(merged_df.info())

# merged_df.to_csv('./wandb_csv_data/merge_data_tings.csv', index=False)
x = merged_df['Step']
columns_to_plot = merged_df.columns.difference(['Step'])
colour_list = ['mediumblue', 'cornflowerblue']
labels = ["R_IPB    Agent 0",
              "R_maxY Agent 1"
              ]
columns_to_plot = columns_to_plot[::-1]
colour_list = colour_list[::-1]
lines = []
for ind, column in enumerate(columns_to_plot[2:]):
    print(column)
    new_df = merged_df[['Step', column]]
    new_df = new_df.dropna()
    y = merged_df[column]
    line, = plt.plot(x, y, label=labels[ind], color=colour_list[ind])
    lines.append(line)


plt.xlabel('Step')
plt.ylabel('Action', labelpad=19)
plt.legend(handles=lines[::-1], labels=["R_IPB    Agent 0", "R_maxY Agent 1"], loc="upper left")
plt.savefig('./wandb_graphs/' + 'maxY_action_comparison' + '.png', bbox_inches="tight", dpi=400)
plt.show()

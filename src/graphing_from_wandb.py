import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pandas as pd
import sys
import numpy as np


# pd.set_option('display.max_columns', None)
# matplotlib.rcParams.update({'font.size': 16})
sns.set(font_scale=0.8)


def read_csv(filename, multi_agent):
    df1 = pd.read_csv("./wandb_csv_data/" + filename)
    if multi_agent:
        base_filename, extension = filename.rsplit('.', 1)
        modified_filename = f"{base_filename[:-1]}1.{extension}"
        df2 = pd.read_csv("./wandb_csv_data/" + modified_filename)

    def process_column(column):
        return column.dropna().head(2000).reset_index(drop=True)  # TODO make this not hardcoded 2000 somehow

    df1 = df1.apply(process_column)
    df1 = df1.drop(['Step'], axis=1)
    if multi_agent:
        df2 = df2.apply(process_column)
        df2 = df2.drop(['Step'], axis=1)
        n = 9  # Number of columns in each slice
        slices_df1 = [df1.iloc[:, i:i + n] for i in range(0, df1.shape[1], n)]
        slices_df2 = [df2.iloc[:, i:i + n] for i in range(0, df2.shape[1], n)]
        interlaced_slices = []
        for slice_df1, slice_df2 in zip(slices_df1, slices_df2):
            interlaced_slices.extend([slice_df1, slice_df2])
        final_df = pd.concat(interlaced_slices, axis=1)

        return final_df

    final_df = df1.iloc[:2000]  # TODO make this not hardcoded 2000 somehow

    return final_df


def moving_avg_graph(df, multi_agent, save_name, title, true_labels, labels):
    if multi_agent:
        colour_list = ['mediumblue',
                       'cornflowerblue',
                       'darkgreen',
                       'mediumseagreen',
                       'rebeccapurple',
                       'plum',
                       'maroon',
                       'darksalmon'
                       # 'black',
                       # 'dimgray',
                       # 'mediumaquamarine',
                       # 'rebeccapurple',
                       ]
    else:
        colour_list = ['cornflowerblue',
                       'darkgreen',
                       'rebeccapurple',
                       'darksalmon',
                       # 'lightgray',
                       # 'forestgreen',
                       # 'mediumaquamarine',
                       # 'rebeccapurple',
                       ]
    all_columns = df.columns
    columns_to_plot = [column for column in all_columns if "MIN" not in column and "MAX" not in column]
    column_groups = [columns_to_plot[i:i + 3] for i in range(0, len(columns_to_plot), 3)]

    value = 0.5

    for ind, group in enumerate(column_groups):
        group_mean = df[group].mean(axis=1)
        group_std = df[group].std(axis=1)
        if true_labels:
            plt.plot(df.index, group_mean, label=group[0], color=colour_list[ind])
        else:
            plt.plot(df.index, group_mean, label=labels[ind], color=colour_list[ind])
        plt.fill_between(df.index, group_mean - value * group_std, group_mean + value * group_std, alpha=0.2,
                         color=colour_list[ind])
    plt.xlabel('Episode')
    plt.ylabel('Moving Average Reward')
    # plt.title(title)
    plt.legend()
    plt.savefig('./wandb_graphs/' + save_name + '.png', bbox_inches="tight", dpi=400)  # dpi=1200)
    plt.show()


if __name__ == "__main__":
    file_name = "homo_agent_only_vs_all_shared_agent_0.csv"
    # multi_agent = False
    multi_agent = True
    save_name = "homo_agent_only_vs_all_shared"
    title = "Moving Average Tings"
    true_labels = False
    # true_labels = True
    labels = ["All Shared Agent 0",
              "All Shared Agent 1",
              "Agent Only Agent 0",
              "Agent Only Agent 1"
              ]

    df = read_csv(file_name, multi_agent)
    moving_avg_graph(df, multi_agent, save_name, title, true_labels, labels)


















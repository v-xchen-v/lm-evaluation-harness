# from autoeval_server.resultparser.loadresults import EvalTaskResultInfo, list_models
from autoeval_server.leaderboard.display import get_leaderboard_df_data
from config import RESULTS_SAVE_ROOT as results_save_root
from eval_and_dumping_result import LEADERBOARDTASK_REGISTRY
import numpy as np
import matplotlib.pyplot as plt
from itertools import chain
import pandas as pd

# get data
def draw(dataset_name="MMLU"):
    df_data = get_leaderboard_df_data(dataset_name=dataset_name, num_fewshot=0, use_cot=False)
    pd_stats = pd.DataFrame.from_dict(df_data)
    pd_stats.to_pickle('board.pkl') 

    pd_stats = pd.read_pickle('board.pkl')

    columns = [
        'next token argmax choices acc',
        'next token argmax all acc',
        'next token argmax choices circular acc',
        'next token argmax all circular acc',
        'acc',
        'acc norm',
        'ppl argmin acc',
    ]


    for col in columns:
        pd_stats[f'rank_{col}'] = pd_stats[f'{col}'].rank(ascending=False)

    rank_columns = [f'rank_{col}' for col in columns]

    cm = plt.get_cmap('gist_rainbow')
    colors = cm(np.linspace(0, 1.0, len(rank_columns)))

    ax1 = pd_stats.plot.scatter(x='Model Name', y=rank_columns[0], label=rank_columns[0], c=colors[0], alpha=0.5)
    ax1 = pd_stats.plot.scatter(x='Model Name', y=rank_columns[1], label=rank_columns[1], c=colors[1], alpha=0.5, ax=ax1)
    ax1 = pd_stats.plot.scatter(x='Model Name', y=rank_columns[2], label=rank_columns[2], c=colors[2], alpha=0.5, ax=ax1)
    ax1 = pd_stats.plot.scatter(x='Model Name', y=rank_columns[3], label=rank_columns[3], c=colors[3], alpha=0.5, ax=ax1)
    ax1 = pd_stats.plot.scatter(x='Model Name', y=rank_columns[4], label=rank_columns[4], c=colors[4], alpha=0.5, ax=ax1)
    ax1 = pd_stats.plot.scatter(x='Model Name', y=rank_columns[5], label=rank_columns[5], c=colors[5], alpha=0.5, ax=ax1)
    ax1 = pd_stats.plot.scatter(x='Model Name', y=rank_columns[6], label=rank_columns[6], c=colors[6], alpha=0.5, ax=ax1)
    ax1.set_ylim(ax1.get_ylim()[::-1])
    plt.xticks(rotation=65)
    plt.xticks(fontsize=8)
    plt.subplots_adjust(bottom=0.15) 
    fig = ax1.get_figure()
    fig.set_size_inches(30, 5)
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.ylabel('rank by metric')

    plt.savefig(f'./output/rank_consistency_{dataset_name}.png')
    
datasets=[
    'MMLU',
    'HellaSwag',
    'ARC-c',
    'ARC-e',
    'BoolQ',
    'PIQA',
    'SIQA',
    'AGIEvalEng',
    
]

for dataset in datasets:
    draw(dataset)

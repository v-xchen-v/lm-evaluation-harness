from autoeval_server.resultparser.loadresults import EvalTaskResultInfo, list_models
from autoeval_server.leaderboard.display import get_leaderboard_df_data
from config import RESULTS_SAVE_ROOT as results_save_root
from eval_and_dumping_result import LEADERBOARDTASK_REGISTRY
import numpy as np
import matplotlib.pyplot as plt
from itertools import chain
import pandas as pd
from collections import defaultdict

llama2_ref = \
[
    {
        "Model": "mosaicml/mpt-7b",
        "MMLU": 26.8,
    },
    {
        "Model": "mosaicml/mpt-30b",
        "MMLU": 46.9,
    },
    {
        "Model": "tiiuae/falcon-7b",
        "MMLU": 26.2,
    },
    {
        "Model": "tiiuae/falcon-40b",
        "MMLU": 55.4,
    },
    {
        "Model": "huggyllama/llama-7b",
        "MMLU": 35.1,
    },
    {
        "Model": "huggyllama/llama-13b",
        "MMLU": 46.9,
    },
    {
        "Model": "huggyllama/llama-30b",
        "MMLU": 57.8,
    },
    {
        "Model": "huggyllama/llama-65b",
        "MMLU": 63.4,
    },
    {
        "Model": "meta-llama/Llama-2-7b-hf",
        "MMLU": 45.3,
    },
    {
        "Model": "meta-llama/Llama-2-13b-hf",
        "MMLU": 54.8,
    },
    {
        "Model": "meta-llama/Llama-2-70b-hf",
        "MMLU": 68.9,
    },
]

columns = [
    'next token argmax choices acc',
    'next token argmax all acc',
    'next token argmax choices circular acc',
    'next token argmax all circular acc',
    'acc',
    'acc norm',
    'ppl argmin acc',
]

metric = columns[-1]
dataset_name = 'HellaSwag'

llama2_ref_modelset = set([item['Model'] for item in llama2_ref])
pd_llama2_ref = pd.DataFrame.from_dict(llama2_ref)
ax1 = pd_llama2_ref.plot.scatter(x='Model', y=dataset_name, label=f'paper {dataset_name}', c='green', alpha=0.5)
df_data = get_leaderboard_df_data(dataset_name=dataset_name, num_fewshot=0, use_cot=False)
pd_stats = pd.DataFrame.from_dict(df_data)
pd_stats = pd_stats[pd_stats['Model Name'].isin(llama2_ref_modelset)]
# pd_stats.to_pickle('board.pkl') 
# pd_stats = pd.read_pickle('board.pkl')


ax1 = pd_stats.plot.scatter(x='Model Name', y=f'{metric}', label='Ours MMLU(0-s)', c='blue', alpha=0.5, ax=ax1)

# Compute and plot average difference
y_diffs = defaultdict()
for modelname in llama2_ref_modelset:
    y1 = pd_llama2_ref.loc[pd_llama2_ref['Model']==modelname, dataset_name].iloc[0]
    y2 = pd_stats.loc[pd_stats['Model Name']==modelname, metric].iloc[0]
    y_diffs[modelname] = y2 - y1

# Annotate each point with its y-coordinate
for i, j in zip(pd_stats['Model Name'], pd_stats[metric]):
    plt.annotate(f'{y_diffs[i]:.1f}', (i, j), textcoords="offset points", xytext=(0,5), ha='center')

plt.xticks(rotation=65)
plt.xticks(fontsize=8)
plt.subplots_adjust(bottom=0.15)
plt.subplots_adjust(bottom=0.15) 
fig = ax1.get_figure()
fig.set_size_inches(10, 5)
plt.ylim(0, 100, 5)
plt.tight_layout()
plt.title(dataset_name)
ax1.figure.savefig(f'{dataset_name}_{"_".join(metric.split(" "))}_ismatch_with_ref.png')
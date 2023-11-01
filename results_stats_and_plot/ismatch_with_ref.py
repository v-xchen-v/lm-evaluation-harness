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
        "HellaSwag": 76.4,
        "ARC-e": 70.2,
        "ARC-c": 42.6,
        "BoolQ": 75.0,
        "PIQA": 80.6,
        "SIQA": 48.5,
        "OpenBookQA_Fact": 51.4,
        "CommonSenseQA": 21.3,
        "AGIEvalEng":23.5,
    },
    {
        "Model": "mosaicml/mpt-30b",
        "MMLU": 46.9,
        "HellaSwag": 79.9,
        "ARC-e": 76.5,
        "ARC-c": 50.6,
        "BoolQ": 79.0,
        "PIQA": 81.9,
        "SIQA": 48.9,
        "OpenBookQA_Fact": 52.0,
        "CommonSenseQA": 58.2,
        "AGIEvalEng":33.8,
    },
    {
        "Model": "tiiuae/falcon-7b",
        "MMLU": 26.2,
        "HellaSwag": 74.1,
        "ARC-e": 70.0,
        "ARC-c": 42.4,
        "BoolQ": 67.5,
        "PIQA": 76.7,
        "SIQA": 47.2,
        "OpenBookQA_Fact": 51.6,
        "CommonSenseQA": 20.8,
        "AGIEvalEng":21.2,
    },
    {
        "Model": "tiiuae/falcon-40b",
        "MMLU": 55.4,
        "HellaSwag": 83.6,
        "ARC-e": 79.2,
        "ARC-c": 54.5,
        "BoolQ": 83.1,
        "PIQA": 82.4,
        "SIQA": 50.1,
        "OpenBookQA_Fact": 56.6,
        "CommonSenseQA": 70.4,
        "AGIEvalEng":37.0,
    },
    {
        "Model": "huggyllama/llama-7b",
        "MMLU": 35.1,
        "HellaSwag": 76.1,
        "ARC-e": 72.8,
        "ARC-c": 47.6,
        "BoolQ": 76.5,
        "PIQA": 79.8,
        "SIQA": 48.9,
        "OpenBookQA_Fact": 57.2,
        "CommonSenseQA": 33.6,
        "AGIEvalEng":23.9,
    },
    {
        "Model": "huggyllama/llama-13b",
        "MMLU": 46.9,
        "HellaSwag": 79.2,
        "ARC-e": 74.8,
        "ARC-c": 52.7,
        "BoolQ": 78.1,
        "PIQA": 80.1,
        "SIQA": 50.4,
        "OpenBookQA_Fact": 56.4,
        "CommonSenseQA": 62.0,
        "AGIEvalEng":33.9,
    },
    {
        "Model": "huggyllama/llama-30b",
        "MMLU": 57.8,
        "HellaSwag": 82.8,
        "ARC-e": 80.0,
        "ARC-c": 57.8,
        "BoolQ": 83.1,
        "PIQA": 82.3,
        "SIQA": 50.4,
        "OpenBookQA_Fact": 58.6,
        "CommonSenseQA": 72.5,
        "AGIEvalEng":41.7,
    },
    {
        "Model": "huggyllama/llama-65b",
        "MMLU": 63.4,
        "HellaSwag": 84.2,
        "ARC-e": 78.9,
        "ARC-c": 56.0,
        "BoolQ": 85.3,
        "PIQA": 82.8,
        "SIQA": 52.3,
        "OpenBookQA_Fact": 60.2,
        "CommonSenseQA": 74.0,
        "AGIEvalEng":47.6,
    },
    {
        "Model": "meta-llama/Llama-2-7b-hf",
        "MMLU": 45.3,
        "HellaSwag": 77.2,
        "ARC-e": 75.2,
        "ARC-c": 45.9,
        "BoolQ": 77.4,
        "PIQA": 78.8,
        "SIQA": 48.3,
        "OpenBookQA_Fact": 58.6,
        "CommonSenseQA": 57.8,
        "AGIEvalEng":23.9,
    },
    {
        "Model": "meta-llama/Llama-2-13b-hf",
        "MMLU": 54.8,
        "HellaSwag": 80.7,
        "ARC-e": 77.3,
        "ARC-c": 49.4,
        "BoolQ": 81.7,
        "PIQA": 80.5,
        "SIQA": 50.3,
        "OpenBookQA_Fact": 57.0,
        "CommonSenseQA": 67.3,
        "AGIEvalEng":39.1,
    },
    {
        "Model": "meta-llama/Llama-2-70b-hf",
        "MMLU": 68.9,
        "HellaSwag": 85.3,
        "ARC-e": 80.2,
        "ARC-c": 57.4,
        "BoolQ": 85.0,
        "PIQA": 82.8,
        "SIQA": 50.7,
        "OpenBookQA_Fact": 60.2,
        "CommonSenseQA": 78.5,
        "AGIEvalEng":54.2,
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

metric = columns[0]
dataset_name = 'AGIEvalEng'
num_fewshot=0

llama2_ref_modelset = set([item['Model'] for item in llama2_ref])
pd_llama2_ref = pd.DataFrame.from_dict(llama2_ref)
ax1 = pd_llama2_ref.plot.scatter(x='Model', y=dataset_name, label=f'paper {dataset_name}', c='green', alpha=0.5)
df_data = get_leaderboard_df_data(dataset_name=dataset_name, num_fewshot=num_fewshot, use_cot=False)
pd_stats = pd.DataFrame.from_dict(df_data)
pd_stats = pd_stats[pd_stats['Model Name'].isin(llama2_ref_modelset)]
print(pd_stats)
# pd_stats.to_pickle('board.pkl') 
# pd_stats = pd.read_pickle('board.pkl')


ax1 = pd_stats.plot.scatter(x='Model Name', y=f'{metric}', label=f'Ours {dataset_name}(0-s)', c='blue', alpha=0.5, ax=ax1)

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
plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.title(dataset_name)
ax1.figure.savefig(f'{dataset_name}_{"_".join(metric.split(" "))}_ismatch_with_ref.png')
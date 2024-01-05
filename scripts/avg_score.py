from autoeval_server.leaderboard.display import DisplayDataset, get_leaderboard_df_data
from autoeval_server.resultparser.getmodelinfo import ModelInfo
from tqdm import tqdm

result_dir = r'/eval_results'

DISPLAY_DATASETS = [
    DisplayDataset(
        dataset_name="MMLU",
        num_fewshot=0,
        use_cot=False,
        abbr="MMLU(0-s)"
    ),
    DisplayDataset(
        dataset_name="HellaSwag",
        num_fewshot=0,
        use_cot=False,
        abbr="HellaSwag(0-s)"
    ),
    DisplayDataset(
        dataset_name="ARC-c",
        num_fewshot=0,
        use_cot=False,
        abbr="ARC-c(0-s)"
    ),
    DisplayDataset(
        dataset_name="ARC-e",
        num_fewshot=0,
        use_cot=False,
        abbr="ARC-e(0-s)"
    ),
    DisplayDataset(
        dataset_name="BoolQ",
        num_fewshot=0,
        use_cot=False,
        abbr="BoolQ(0-s)"
    ),
    DisplayDataset(
        dataset_name="PIQA",
        num_fewshot=0,
        use_cot=False,
        abbr="PIQA(0-s)"
    ),
    DisplayDataset(
        dataset_name="SIQA",
        num_fewshot=0,
        use_cot=False,
        abbr="SIQA(0-s)"
    ),
    DisplayDataset(
        dataset_name="AGIEvalEng",
        num_fewshot=0,
        use_cot=False,
        abbr="AGIEvalEng(0-s)"
    ),
    DisplayDataset(
        dataset_name="OpenBookQA_Fact",
        num_fewshot=0,
        use_cot=False,
        abbr="OpenBookQA with Fact(0-s)"
    ),
    DisplayDataset(
        dataset_name="CommonSenseQA",
        num_fewshot=0,
        use_cot=False,
        abbr="CommonSenseQA(0-s)"
    ),
    DisplayDataset(
        dataset_name="RACE_all",
        num_fewshot=0,
        use_cot=False,
        abbr="RACE_all(0-s)"
    )
]

models = [
    'WizardLM/WizardLM-7B-V1.0',
    'WizardLM/WizardLM-13B-V1.2',
    'WizardLM/WizardLM-70B-V1.0',
    'Xwin-LM/Xwin-LM-7B-V0.1', 
    'Xwin-LM/Xwin-LM-13B-V0.1',
    # 'Xwin-LM/Xwin-LM-70B-V0.1',
    'chavinlo/alpaca-native', 
    'chavinlo/alpaca-13b', 
    'huggyllama/llama-7b', 
    'huggyllama/llama-13b', 
    'huggyllama/llama-30b', 
    'huggyllama/llama-65b', 
    'lmsys/vicuna-7b-v1.5', 
    'lmsys/vicuna-13b-v1.5',
    'lmsys/vicuna-33b-v1.3', 
    'meta-llama/Llama-2-7b-chat-hf', 
    'meta-llama/Llama-2-13b-chat-hf', 
    'meta-llama/Llama-2-70b-chat-hf', 
    'meta-llama/Llama-2-7b-hf', 
    'meta-llama/Llama-2-13b-hf', 
    'meta-llama/Llama-2-70b-hf', 
    'mosaicml/mpt-7b', 
    'mosaicml/mpt-30b', 
    'mosaicml/mpt-30b-chat', 
    'tiiuae/falcon-7b',
    'tiiuae/falcon-40b', 
    ]
custom_eval_metrics = [
    "acc",
    "acc_norm",
    "ppl_argmin_acc",
    "next_token_argmax_choices_acc",
    "next_token_argmax_all_acc",
    "next_token_argmax_choices_circular_acc",
    "next_token_argmax_all_circular_acc"
]
dataset_names = [x.dataset_name for x in DISPLAY_DATASETS]

def parse_data(dataset_names, model_names, metric_names):
    # only consider 0-s
    num_fewshot = 0
    
    parsed_data = []
    for dataset_name in tqdm(dataset_names):
        dataset_data = get_leaderboard_df_data(dataset_name, num_fewshot, False)
        for model_name in model_names:
            for metric_name in metric_names:
                metric_value = [x for x in dataset_data if x['Model Name']==model_name][0][metric_name.replace('_', ' ')]
                parsed_data.append({
                    'dataset': dataset_name,
                    'model': model_name,
                    'metric': metric_name,
                    'value': metric_value,
                })
    return parsed_data
    # print(parsed_data)
    
import numpy as np
import pandas as pd
def avg_metric_across_dataset(model_names, metric_names, parsed_data):
    avg_metrics = []
    for model_name in model_names:
        for metric_name in metric_names:
            metric_list = [x['value'] for x in parsed_data if x['metric']==metric_name and x['model']==model_name]
            metric_avg = np.average(np.array(metric_list)) 
            avg_metrics.append({
                'model': model_name,
                'metric': metric_name,
                'avg': metric_avg
            })      
    pd.DataFrame.from_dict(avg_metrics).to_csv('avg.csv')
avg_metric_across_dataset(models, custom_eval_metrics, parse_data(dataset_names, models, custom_eval_metrics))
    

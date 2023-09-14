from autoeval_server.resultparser.loadresults import list_models, get_leaderboard_aggregated_metrics, get_tasks_aggregated_metrics
from eval_and_dumping_result import LEADERBOARDTASK_REGISTRY
from autoeval_server.leaderboard.displayutils import make_clickable_model
from leaderboardtask import LeaderBoardTask
from dataclasses import dataclass
import numpy as np

@dataclass
class DisplayDataset:
    # dataset name, one dataset could have multiple eval tasks to cpntruct different requests and get multiple metrics
    dataset_name: str
    
    # the fewshot number
    num_fewshot:int
    
    # using cot or not, for AGIEval benchmark
    use_cot: bool
    
    # the display name
    abbr: str
    
DISPLAY_DATASETS = [ \
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
]

# function that filters eval tasks
# def filter_evaltask(evaltask: LeaderBoardTask, dataset_name: str, num_fewshot:int, use_cot: bool):
#     return evaltask.dataset_name == dataset_name and evaltask.num_fewshot == num_fewshot and evaltask.use_cot == use_cot
# # using filter function to filter eval task sequence by dataset setting
# filtered = filter(filter_evaltask, LEADERBOARDTASK_REGISTRY.values())   
     
def get_leaderboard_df_data(dataset_name: str, num_fewshot: int, use_cot: bool):
    """
    parse result folder, find all models with results in it and parse to df
    returns: list(dict)
    
    MMLU:
    -     | metric1 | metric2 |
    model |
    
    HellaSwag:
    -     | metric1 | metric2 |
    model |
    """
    models = list_models()
    leaderboard_df_data = []
    for model_name in models:
        row_data_dict = \
        {
            "Model Name": make_clickable_model(model_name),
        }
        
        # filter out the dataset with settings related eval tasks
        display_evaltasks = list(filter(lambda x: x.dataset_name == dataset_name and x.num_fewshot == num_fewshot and x.use_cot == use_cot, LEADERBOARDTASK_REGISTRY.values()))
        aggregate_metrics = get_tasks_aggregated_metrics(model_name, display_evaltasks)
        for metric_name, aggregated_metricval in aggregate_metrics.items():
            if aggregated_metricval is not None:
                aggregated_metricval = np.round(aggregated_metricval*100,1)
            row_data_dict[metric_name.replace('_', ' ')]= aggregated_metricval
        leaderboard_df_data.append(row_data_dict)
    return leaderboard_df_data


if __name__ == "__main__":
    pass
    # df_data = get_leaderboard_df_data("MMLU", 0, False)
    # print(df_data)
    # import pandas as pd
    # df = pd.DataFrame.from_records(df_data)
    # sortedDf = df.sort_index(axis=1)
    # print(sortedDf)
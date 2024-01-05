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

# pick some dataset want to show, the others will be hidden.   
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
        dataset_name="Winogrande-ori",
        num_fewshot=0,
        use_cot=False,
        abbr="Winogrande-ori(0-s)"
    ),
    DisplayDataset(
        dataset_name="Winogrande",
        num_fewshot=0,
        use_cot=False,
        abbr="Winogrande(0-s)"
    ),
    DisplayDataset(
        dataset_name="SIQA",
        num_fewshot=0,
        use_cot=False,
        abbr="SIQA(0-s)"
    ),
    DisplayDataset(
        dataset_name="AGIEvalEng",
        num_fewshot=5,
        use_cot=False,
        abbr="AGIEvalEng ori(3-5-s)"
    ),
    DisplayDataset(
        dataset_name="AGIEvalEng",
        num_fewshot=0,
        use_cot=False,
        abbr="AGIEvalEng(0-s)"
    ),
    DisplayDataset(
        dataset_name="OpenBookQA",
        num_fewshot=0,
        use_cot=False,
        abbr="OpenBookQA(0-s)"
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
        dataset_name="RACE_high",
        num_fewshot=0,
        use_cot=False,
        abbr="RACE_high(0-s)"
    ),
    DisplayDataset(
        dataset_name="RACE_middle",
        num_fewshot=0,
        use_cot=False,
        abbr="RACE_middle(0-s)"
    ),
    DisplayDataset(
        dataset_name="RACE_all",
        num_fewshot=0,
        use_cot=False,
        abbr="RACE_all(0-s)"
    ),
    DisplayDataset(
        dataset_name="CommonSenseQA",
        num_fewshot=0,
        use_cot=False,
        abbr="CommonSenseQA(7-s)"
    ),
    DisplayDataset(
        dataset_name="MMLU",
        num_fewshot=5,
        use_cot=False,
        abbr="MMLU(5-s)"
    ),
    DisplayDataset(
        dataset_name="HellaSwag",
        num_fewshot=10,
        use_cot=False,
        abbr="HellaSwag(10-s)"
    ),
    DisplayDataset(
        dataset_name="ARC-c",
        num_fewshot=25,
        use_cot=False,
        abbr="ARC(25-s)"
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
            # "Model Name": make_clickable_model(model_name),
            "Model Name": model_name,
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
    # pass
    # import cProfile
    # cProfile.run('get_leaderboard_df_data("MMLU", 5, False)')
    # dataset_name='MMLU'
    # dataset_name='HellaSwag'
    # dataset_name='ARC-c'
    # dataset_name="ARC-e"
    # dataset_name="BoolQ"
    # dataset_name="SIQA"
    # dataset_name="PIQA"
    # dataset_name = 'AGIEvalEng'
    # dataset_name = 'OpenBookQA'
    # dataset_name = 'OpenBookQA_Fact'
    # dataset_name = 'CommonSenseQA'
    dataset_name = 'RACE-all'
    num_fewshot=0
    df_data = get_leaderboard_df_data(dataset_name, num_fewshot, False)
    print(df_data)
    
    import csv

    # to_csv = [
    #     {'name': 'bob', 'age': 25, 'weight': 200},
    #     {'name': 'jim', 'age': 31, 'weight': 180},
    # ]

    keys = df_data[0].keys()

    with open(f'./results/leaderboard/{dataset_name}_{num_fewshot}s.csv', 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(df_data)
    
    # import pandas as pd
    # df = pd.DataFrame.from_records(df_data)
    # sortedDf = df.sort_index(axis=1)
    # print(sortedDf)
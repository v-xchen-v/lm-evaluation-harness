from resultparser.loadresults import list_models, is_result_exists, get_leaderboard_aggregated_metric
from config import LEADERBOARDTASK_REGISTRY
from leaderboard.displayutils import make_clickable_model

def get_leaderboard_df_data():
    """
    parse result folder, find all models with results in it and parse to df
    returns: list(dict)
    """
    models = list_models()
    leaderboard_df_data = []
    for model in models:
        row_data_dict = \
        {
            "Model Name": make_clickable_model(model),
        }
        for task_abbr, task in LEADERBOARDTASK_REGISTRY.items():
            if is_result_exists(model, task.name, task.task_version, task.num_fewshot):
                row_data_dict[task.abbr]= get_leaderboard_aggregated_metric(model, task)
            else:
                # pass
                print(f"no result for {model} {task_abbr}")
        leaderboard_df_data.append(row_data_dict)
    return leaderboard_df_data


if __name__ == "__main__":
    df_data = get_leaderboard_df_data()
    print(df_data)
    import pandas as pd
    df = pd.DataFrame.from_records(get_leaderboard_df_data())
    sortedDf = df.sort_index(axis=1)
    print(sortedDf)
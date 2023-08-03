import json
from resultparser.getmodekinfo import model_info
from typing import Tuple
from config import results_save_root

# from dataclasses import dataclass
# # load eval result to memory dictionary
from pathlib import Path
import numpy as np
import os

# from auto_leaderboard.leaderboard_tasks import LEADERBOARDTASK_REGISTRY

def list_subdirectories_by_level(directory, slevel=1):
    """
    list the subdirectories by level.
    """

    if not Path(directory).is_dir():
        raise Exception(f"The directory not exists")
    
    if slevel < 1:
         raise "slevel should bigger than 1"
    subdirectories = []
    stack = [(directory, 0)]

    while True:
        current_dir, level = stack.pop()
        if level == slevel-1:
            for item_path in Path(current_dir).iterdir():
                  if Path(item_path).is_dir():
                    subdirectories.append(item_path)
            break
        
        if level < slevel-1:
            for item_path in Path(current_dir).iterdir():
                    stack.append((item_path, level + 1))

    return subdirectories

def list_models():
    """
    list tasks have results
    """
    return [model_info.get_modelname(Path(dirname).name) for dirname in list_subdirectories_by_level(results_save_root, 1) if len(Path(dirname).name) == len("1b1847529b6cccf732847cb2120f7ef0") and len(list_subdirectories_by_level(results_save_root, 1))>=1]

def list_tasks(model_name):
    return [Path(dirname).name for dirname in list_subdirectories_by_level(Path(results_save_root)/model_info.get_hashed_modelname(model_name), 1)]

def list_tasks_version(model_name, tasks_name):
    return [Path(dirname).name for dirname in list_subdirectories_by_level(Path (results_save_root)/model_info.get_hashed_modelname(model_name)/tasks_name, 1)]



def get_lastest_tasks_version(model_name, tasks_name):
    task_dir = Path (results_save_root)/model_info.get_hashed_modelname(model_name)/tasks_name
    if Path(task_dir).is_dir() and len(list_subdirectories_by_level(task_dir, 1))>=1:
        return max(map(int, [Path(dirname).name for dirname in list_subdirectories_by_level(task_dir, 1)]))
    else:
        return None

def list_subtasks_result_jsons(model_name, tasks_name, num_fewshot):
    dirname = Path(results_save_root)/model_info.get_hashed_modelname(model_name)/tasks_name
    tasks_version = get_lastest_tasks_version(model_name, tasks_name)
    if tasks_version is not None:
        dirname = dirname/str(tasks_version)/f"{num_fewshot}shot"
        return list(dirname.glob(f"results.json"))
    return None

def get_subtask_latestversion_result_jsonpath(model_name, task_name, num_fewshot):
    dirname = Path(results_save_root)/model_info.get_hashed_modelname(model_name)/task_name
    tasks_version = get_lastest_tasks_version(model_name, task_name)
    jsonpath = dirname/str(tasks_version)/f"{num_fewshot}shot"/'results.json'
    return jsonpath

def is_lastestversion_result_exists(model_name, task_name, num_fewshot):
    return Path(get_subtask_latestversion_result_jsonpath(model_name, task_name, num_fewshot)).is_file()

def get_subtask_result_jsonpath(model_name, task_name, version, num_fewshot):
    dirname = Path(results_save_root)/model_info.get_hashed_modelname(model_name)/task_name
    jsonpath = dirname/str(version)/f"{num_fewshot}shot"/'results.json'
    return jsonpath

def is_result_exists(model_name, task_name, version, num_fewshot):
    return Path(get_subtask_result_jsonpath(model_name, task_name, version,num_fewshot)).is_file()

def parse_tasks_eval_results(json_filepath: str):
    """
    parse result per task
    returns: a dict contains subtasks result
    """
    with open(json_filepath, 'r') as f:
        # dict of subtask name to dict of metricname to value
        results = json.load(f)['results']

    return results

def get_metric(model_name, task_name, num_fewshot, use_cot, subtask_name, metric_name):
    hashed_modelname = model_info.get_hashed_modelname(model_name)
    task_save_root = os.path.join(results_save_root,hashed_modelname,task_name)
    result_jsonpath = os.path.join(task_save_root,str(get_lastest_tasks_version(model_name, task_name)),f'{num_fewshot}shot', 'results.json')
    return parse_tasks_eval_results(result_jsonpath)[subtask_name][metric_name]

def get_leaderboard_metrics(model_name, leaderboard_task):
    metrics =[]
    for subtask in leaderboard_task.subtasks:
        metric = get_metric(model_name, leaderboard_task.name, leaderboard_task.num_fewshot, leaderboard_task.use_cot, subtask, leaderboard_task.metric)
        metrics.append(metric)
    return metrics

def get_leaderboard_aggregated_metric(model_name, leaderboard_task):
    metrics = get_leaderboard_metrics(model_name, leaderboard_task)
    if leaderboard_task.aggregate_op == "mean":
        average = np.average(metrics)
        return np.round(average*100,1)
    else:
        raise Exception("unsupport aggregate operation")

def get_subtask_write_out_info(model_name, task_name, subtask_name, num_fewshot):
    hashed_modelname = model_info.get_hashed_modelname(model_name)
    task_save_root = os.path.join(results_save_root,hashed_modelname,task_name)
    writeout_infofile = os.path.join(task_save_root,str(get_lastest_tasks_version(model_name, task_name)),f'{num_fewshot}shot', f'{subtask_name}_write_out_info.json')

    return open(writeout_infofile, 'r').readlines()






# # @dataclass
# # class EvalResult:
# #     task: str
# #     # eval_name: str
# #     # evaluator_version: str
# #     model: str
# #     # model_org: str
# #     # model_revision: str
# #     result: dict
# task_names= ['truthfulqa_mc']
# taskname2metrics = {'truthfulqa_mc':['mc2']}
# def load_results(file_path):
#     file_path = r'C:\Users\xichen6\Documents\CodeSpace\eval_board\eval_results\fc433f70103338181ac914a44eb2749c\truthfulqa\1\0shot\results.json'
#     with open(file_path, 'r') as f:
#         results = json.load(f)
#     print(results['results'])
#     for task_name in task_names:
#         for metric_name in taskname2metrics[task_name]:
#             print(f'{task_name} {metric_name}: {results["results"][task_name][metric_name]}')
# load_results(None)
if __name__ == "__main__":
    print(list_models())
    print(list_tasks('huggyllama/llama-7b'))
    print(list_tasks_version('huggyllama/llama-7b', 'truthfulqa'))
    print(get_lastest_tasks_version('huggyllama/llama-7b', 'truthfulqa'))
    print(list_subtasks_result_jsons('huggyllama/llama-7b', 'truthfulqa', num_fewshot=0))

    # # get results
    # print(aggregate_eval_results('huggyllama/llama-7b',"mmlu", num_fewshot=5))
    # print(aggregate_eval_results('huggyllama/llama-7b',"truthfulqa", num_fewshot=0))
    # # print(aggregate_eval_results('huggyllama/llama-7b',"hellaswag", num_fewshot=10))
    # print(aggregate_eval_results('huggyllama/llama-7b',"arc", num_fewshot=25))

    # get console out
    # print(get_console_out('huggyllama/llama-7b',"mmlu", num_fewshot=5))

    print(get_metric('huggyllama/llama-7b', 'arc', 25, False, 'arc_challenge', 'acc_norm'))
    # print(get_leaderboard_metrics('huggyllama/llama-7b', LEADERBOARDTASK_REGISTRY[0]))
    # print(get_leaderboard_aggregated_metric('huggyllama/llama-7b', LEADERBOARDTASK_REGISTRY[0]))


from resultparser.loadresults import is_result_exists, list_models
from config import leaderboard_tasks
from evalrunner.evaltask import EvalModelTask
from evalrunner.taskqueue import task_queue

def get_pending_evaluations():
    return task_queue.list_pending_tasks()

def get_finished_evaluations():
    return get_manually_finished_evaluations()

def get_manually_finished_evaluations():
    manually_finished_evaluations = []
    for model in list_models():
        for leaderboard_task in leaderboard_tasks:
            finished = is_result_exists(model_name=model, task_name=leaderboard_task.name, version=leaderboard_task.task_version, num_fewshot=leaderboard_task.num_fewshot)
            if finished:
                manually_finished_evaluations.append(EvalModelTask(model=model, eval_task=leaderboard_task))
    return manually_finished_evaluations

def get_running_evaluations():
    if task_queue.processing_task is None:
        return []
    return [task_queue.processing_task]
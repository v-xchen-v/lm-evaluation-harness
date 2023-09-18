from autoeval_server.resultparser.loadresults import list_models, EvalTaskResultInfo
from eval_and_dumping_result import LEADERBOARDTASK_REGISTRY
from autoeval_server.evalrunner.evaltask import EvalModelTask
from autoeval_server.evalrunner.taskqueue import task_queue
from config import RESULTS_SAVE_ROOT

def get_pending_evaluations():
    return task_queue.list_pending_tasks()

def get_finished_evaluations():
    return _get_manually_finished_evaluations()

def _get_manually_finished_evaluations():
    manually_finished_evaluations = []
    for model_name in list_models():
        for leaderboard_task in LEADERBOARDTASK_REGISTRY.values():
            finished = EvalTaskResultInfo.from_evaltask(results_save_root=RESULTS_SAVE_ROOT, model_id=model_name, leaderboardtask=leaderboard_task).is_result_exists()
            if finished:
                manually_finished_evaluations.append(EvalModelTask(model=model_name, eval_task=leaderboard_task))
    return manually_finished_evaluations

def get_running_evaluations():
    if task_queue.processing_task is None:
        return []
    return [task_queue.processing_task]

if __name__ == '__main__':
    # print(get_finished_evaluations())
    print(get_pending_evaluations())
    print(get_running_evaluations())
    pass
     
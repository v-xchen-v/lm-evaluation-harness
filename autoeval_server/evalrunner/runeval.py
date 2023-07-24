import time
from evalrunner.evaltask import EvalModelTask

def run_eval(eval_model_task: EvalModelTask):
    print(f"[INFO] starting evaluation of {eval_model_task.model} on {eval_model_task.eval_task}")
    time.sleep(50)
    print(f"[INFO] completed evaluation f {eval_model_task.model} on {eval_model_task.eval_task}")
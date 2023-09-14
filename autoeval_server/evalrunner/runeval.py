"""Eval tasks on local machine of model parallel"""

import time
from evalrunner.evaltask import EvalModelTask
import argparse
from eval_and_dumping_result import eval_and_dump
from eval_and_dumping_result import LEADERBOARDTASK_REGISTRY
# sys.path.insert(0, "/repos/lm-evaluation-harness")

from config import results_save_root

def run_eval(eval_model_task: EvalModelTask):
    print(f"[INFO] starting evaluation of {eval_model_task.model} on {eval_model_task.eval_task.abbr}")
    
    # time.sleep(5)
    call_eval(eval_model_task)

    print(f"[INFO] completed evaluation f {eval_model_task.model} on {eval_model_task.eval_task.abbr}")

def parse_args(eval_model_task: EvalModelTask):
    parser = argparse.ArgumentParser()
    parser.add_argument("--leaderboard_task", required=True, choices=LEADERBOARDTASK_REGISTRY, default=eval_model_task.eval_task.name)
    parser.add_argument("--hf_model_name", type=str, required=True, default=eval_model_task.model)
    parser.add_argument("--batch_size", type=str, default="auto")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--limit", type=float, default=None,
                        help="Limit the number of examples per task. "
                             "If <1, limit is a percentage of the total number of examples.")
    parser.add_argument("--num_fewshot", type=int, default=eval_model_task.eval_task.num_fewshot)
    
    if eval_model_task.eval_task.name.find("circular")!=-1:
        use_cache=False
    else:
        use_cache=True
    parser.add_argument("--no_cache", action="store_true", default=use_cache)
    parser.add_argument("--use_data_parallel", action="store_true", default=False)
    parser.add_argument("--use_model_parallel", action="store_true", default=True)
    parser.add_argument("--output_base_path", type=str, default=f"{results_save_root}")
    return parser.parse_args()

def call_eval(eval_model_task: EvalModelTask):
    args = parse_args()

    if args.limit:
        print(
            "WARNING: --limit SHOULD ONLY BE USED FOR TESTING. REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT."
        )

    if args.use_data_parallel and args.use_model_parallel:
        raise(
            "not support use model parallel and data parallel together."
        )
    
    print(f"Selected Leader Board Task: {args.leaderboard_task}")
    print(f"args: {args}")
    
    eval_and_dump(
        leaderboardtask_name=args.leaderboard_task,
        hf_model_name=args.hf_model_name,
        batch_size=args.batch_size,
        device=args.device,
        no_cache=args.no_cache,
        limit=args.limit,
        use_data_parallel=args.use_data_parallel,
        use_model_parallel=args.use_model_parallel,
        num_fewshot=args.num_fewshot,
        output_base_path=args.output_base_path,
    )
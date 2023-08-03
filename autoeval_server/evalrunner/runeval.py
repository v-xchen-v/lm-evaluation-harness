import time
from evalrunner.evaltask import EvalModelTask
import argparse
import json
import logging
import os
import sys
sys.path.insert(0, "/repos/lm-evaluation-harness")

from lm_eval import tasks, evaluator, utils
from config import results_save_root
from resultparser.getmodekinfo import model_info
from pathlib import Path

def run_eval(eval_model_task: EvalModelTask):
    print(f"[INFO] starting evaluation of {eval_model_task.model} on {eval_model_task.eval_task.abbr}")
    
    model_info.add_model(model_name=eval_model_task.model)
    # time.sleep(5)
    call_eval(eval_model_task)

    print(f"[INFO] completed evaluation f {eval_model_task.model} on {eval_model_task.eval_task.abbr}")

def parse_args(eval_model_task: EvalModelTask):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="hf-causal-experimental")
    parser.add_argument("--model_args", default=f"pretrained={eval_model_task.model}")
    parser.add_argument("--tasks", default=f'{",".join(eval_model_task.eval_task.subtasks)}', choices=utils.MultiChoice(tasks.ALL_TASKS))
    parser.add_argument("--provide_description", action="store_true")
    parser.add_argument("--num_fewshot", type=int, default=f'{eval_model_task.eval_task.num_fewshot}')
    parser.add_argument("--batch_size", type=str, default=None)
    parser.add_argument("--max_batch_size", type=int, default=None,
                        help="Maximal batch size to try with --batch_size auto")
    parser.add_argument("--device", type=str, default=None)
    result_filename = "results.json"
    output_dir = Path(results_save_root)/model_info.get_hashed_modelname(eval_model_task.model)/eval_model_task.eval_task.name/str(eval_model_task.eval_task.task_version)/f'{eval_model_task.eval_task.num_fewshot}shot'
    # os.makedirs(output_dir, exist_ok=True)
    parser.add_argument("--output_path", default=Path(output_dir)/result_filename)
    parser.add_argument("--limit", type=float, default=None,
                        help="Limit the number of examples per task. "
                             "If <1, limit is a percentage of the total number of examples.")
    parser.add_argument("--data_sampling", type=float, default=None)
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--decontamination_ngrams_path", default=None)
    parser.add_argument("--description_dict_path", default=None)
    parser.add_argument("--check_integrity", action="store_true")
    parser.add_argument("--write_out", action="store_true", default=True)
    parser.add_argument("--output_base_path", type=str, default=output_dir)

    return parser.parse_args()

def call_eval(eval_model_task: EvalModelTask):
    args = parse_args(eval_model_task)

    assert not args.provide_description  # not implemented

    if args.limit:
        print(
            "WARNING: --limit SHOULD ONLY BE USED FOR TESTING. REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT."
        )

    if args.tasks is None:
        task_names = tasks.ALL_TASKS
    else:
        task_names = utils.pattern_match(args.tasks.split(","), tasks.ALL_TASKS)

    print(f"Selected Tasks: {task_names}")

    description_dict = {}
    if args.description_dict_path:
        with open(args.description_dict_path, "r") as f:
            description_dict = json.load(f)

    results = evaluator.simple_evaluate(
        model=args.model,
        model_args=args.model_args,
        tasks=task_names,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        max_batch_size=args.max_batch_size,
        device=args.device,
        no_cache=args.no_cache,
        limit=args.limit,
        description_dict=description_dict,
        decontamination_ngrams_path=args.decontamination_ngrams_path,
        check_integrity=args.check_integrity,
        write_out=args.write_out,
        output_base_path=args.output_base_path,
    )

    dumped = json.dumps(results, indent=2)
    print(dumped)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
        f.write(dumped)

    batch_sizes = ",".join(map(str, results["config"]["batch_sizes"]))
    print(
        f"{args.model} ({args.model_args}), limit: {args.limit}, provide_description: {args.provide_description}, "
        f"num_fewshot: {args.num_fewshot}, batch_size: {args.batch_size}{f' ({batch_sizes})' if batch_sizes else ''}"
    )
    print(evaluator.make_table(results))
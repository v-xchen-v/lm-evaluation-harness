import argparse
import json
import logging
import os

from lm_eval import tasks, evaluator, utils
from lm_eval.tasks.hendrycks_test import SUBJECTS as MMLU_SUBJECTS
from leaderboardtask import LeaderBoardTask

logging.getLogger("openai").setLevel(logging.WARNING)

def hash_md5(model_name: str):
    import hashlib
    return hashlib.md5(model_name.encode('utf-8')).hexdigest()

custom_eval_tasktypes=[ \
    "likelihoodoptioncontent",
    "likelihoodoptionkeycircular", 
    "greedyoptionkey", 
    "greedyanswer"
]

# TODO：supports multiple metrics insteads of single metric now.
custom_eval_metrics=[ \
    ["ppl_argmax_acc"], 
    ["next_token_argmax_choices_acc", 
    "next_token_argmax_all_acc",
    "next_token_argmax_choices_circular_acc",
    "next_token_argmax_all_circular_acc"],
    ["greedy_is_exact_match_acc"], 
    ["greedy_is_gpt4_match_acc"],
]

custom_eval_metric_aggregate_ops=[\
    ["mean"],
    ["mean", "mean", "mean", "mean"],
    ["mean"],
    ["mean"]    
]

# TODO: as config file
leaderboard_tasks = [
    # LeaderBoardTask(
    #     name="mmlu",
    #     abbr="MMLU(5 shot)",
    #     num_fewshot=5,
    #     use_cot=False,
    #     subtasks= [f'hendrycksTest-{sub}' for sub in  MMLU_SUBJECTS],
    #     metrics=["acc"],
    #     aggregate_ops=['mean'],
    #     version=1,
    #     dataset_name="MMLU",
    # ),
    LeaderBoardTask(
        name=f"mmlu_{custom_eval_tasktypes[0]}",
        abbr="MMLU Option Content(0 shot)",
        num_fewshot=0,
        use_cot=False,
        subtasks= [f"hendrycksTest-{sub}_{custom_eval_tasktypes[0]}" for sub in MMLU_SUBJECTS],
        metrics=custom_eval_metrics[0],
        aggregate_ops=custom_eval_metric_aggregate_ops[0],
        version=0,
        dataset_name="MMLU",
    ),
    LeaderBoardTask(
        name=f"mmlu_{custom_eval_tasktypes[1]}",
        abbr="MMLU Option Key Circular(0 shot)",
        num_fewshot=0,
        use_cot=False,
        subtasks= [f"hendrycksTest-{sub}_{custom_eval_tasktypes[1]}" for sub in MMLU_SUBJECTS],
        metrics=custom_eval_metrics[1],
        aggregate_ops=custom_eval_metric_aggregate_ops[1],
        version=0,
        dataset_name="MMLU",
    ),
    LeaderBoardTask(
        name=f"mmlu_{custom_eval_tasktypes[2]}",
        abbr="MMLU Greedy Option Key(0 shot)",
        num_fewshot=0,
        use_cot=False,
        subtasks= [f"hendrycksTest-{sub}_{custom_eval_tasktypes[2]}" for sub in MMLU_SUBJECTS],
        metrics=custom_eval_metrics[2],
        aggregate_ops=custom_eval_metric_aggregate_ops[2],
        version=0,
        dataset_name="MMLU",
    ),
    LeaderBoardTask(
        name=f"mmlu_{custom_eval_tasktypes[3]}",
        abbr="MMLU Greedy Answer(0 shot)",
        num_fewshot=0,
        use_cot=False,
        subtasks= [f"hendrycksTest-{sub}_{custom_eval_tasktypes[3]}" for sub in MMLU_SUBJECTS],
        metrics=custom_eval_metrics[3],
        aggregate_ops=custom_eval_metric_aggregate_ops[3],
        version=0,
        dataset_name="MMLU",
    ),
       LeaderBoardTask(
        name=f"mmlu_{custom_eval_tasktypes[0]}_5s",
        abbr="MMLU Option Content(5 shot)",
        num_fewshot=5,
        use_cot=False,
        subtasks= [f"hendrycksTest-{sub}_{custom_eval_tasktypes[0]}" for sub in MMLU_SUBJECTS],
        metrics=custom_eval_metrics[0],
        aggregate_ops=custom_eval_metric_aggregate_ops[0],
        version=0,
        dataset_name="MMLU",
    ),
    LeaderBoardTask(
        name=f"mmlu_{custom_eval_tasktypes[1]}_5s",
        abbr="MMLU Option Key Circular(5 shot)",
        num_fewshot=5,
        use_cot=False,
        subtasks= [f"hendrycksTest-{sub}_{custom_eval_tasktypes[1]}" for sub in MMLU_SUBJECTS],
        metrics=custom_eval_metrics[1],
        aggregate_ops=custom_eval_metric_aggregate_ops[1],
        version=0,
        dataset_name="MMLU",
    ),
    LeaderBoardTask(
        name=f"mmlu_{custom_eval_tasktypes[2]}_5s",
        abbr="MMLU Greedy Option Key(5 shot)",
        num_fewshot=5,
        use_cot=False,
        subtasks= [f"hendrycksTest-{sub}_{custom_eval_tasktypes[2]}" for sub in MMLU_SUBJECTS],
        metrics=custom_eval_metrics[2],
        aggregate_ops=custom_eval_metric_aggregate_ops[2],
        version=0,
        dataset_name="MMLU",
    ),
    LeaderBoardTask(
        name=f"mmlu_{custom_eval_tasktypes[3]}_5s",
        abbr="MMLU Greedy Answer(5 shot)",
        num_fewshot=5,
        use_cot=False,
        subtasks= [f"hendrycksTest-{sub}_{custom_eval_tasktypes[3]}" for sub in MMLU_SUBJECTS],
        metrics=custom_eval_metrics[3],
        aggregate_ops=custom_eval_metric_aggregate_ops[3],
        version=0,
        dataset_name="MMLU",
    ),
    # LeaderBoardTask(
    #     name="truthfulqa",
    #     abbr="TruthfulQA(0 shot)",
    #     num_fewshot=0,
    #     use_cot = False,
    #     subtasks=["truthfulqa_mc"],
    #     metrics = ["mc2"],
    #     aggregate_ops=['mean'],
    #     version=1
    # ),
    # LeaderBoardTask(
    #     name="hellaswag",
    #     abbr="HellaSwag(10 shot)",
    #     num_fewshot=10,
    #     use_cot=False,
    #     subtasks=["hellaswag"],
    #     metrics=["acc_norm"],
    #     aggregate_ops=['mean'],
    #     version=0,
    # ),
    LeaderBoardTask(
        name=f"hellaswag_{custom_eval_tasktypes[0]}",
        abbr="HellaSwag Option Content(0 shot)",
        num_fewshot=0,
        use_cot=False,
        subtasks=[f"hellaswag_{custom_eval_tasktypes[0]}"],
        metrics=custom_eval_metrics[0],
        aggregate_ops=custom_eval_metric_aggregate_ops[0],
        version=0,
        dataset_name="HellaSwag",
    ),
    LeaderBoardTask(
        name=f"hellaswag_{custom_eval_tasktypes[1]}",
        abbr="HellaSwag Option Key Circular(0 shot)",
        num_fewshot=0,
        use_cot=False,
        subtasks=[f"hellaswag_{custom_eval_tasktypes[1]}"],
        metrics=custom_eval_metrics[1],
        aggregate_ops=custom_eval_metric_aggregate_ops[1],
        version=0,
        dataset_name="HellaSwag",
    ),
    LeaderBoardTask(
        name=f"hellaswag_{custom_eval_tasktypes[2]}",
        abbr="HellaSwag Greedy Option Key(0 shot)",
        num_fewshot=0,
        use_cot=False,
        subtasks=[f"hellaswag_{custom_eval_tasktypes[2]}"],
        metrics=custom_eval_metrics[2],
        aggregate_ops=custom_eval_metric_aggregate_ops[2],
        version=0,
        dataset_name="HellaSwag",
    ),
    LeaderBoardTask(
        name=f"hellaswag_{custom_eval_tasktypes[3]}",
        abbr="HellaSwag Answer(0 shot)",
        num_fewshot=0,
        use_cot=False,
        subtasks=[f"hellaswag_{custom_eval_tasktypes[3]}"],
        metrics=custom_eval_metrics[3],
        aggregate_ops=custom_eval_metric_aggregate_ops[3],
        version=0,
        dataset_name="HellaSwag",
    ),
    
    LeaderBoardTask(
        name=f"hellaswag_{custom_eval_tasktypes[0]}_10s",
        abbr="HellaSwag Option Content(10 shot)",
        num_fewshot=10,
        use_cot=False,
        subtasks=[f"hellaswag_{custom_eval_tasktypes[0]}"],
        metrics=custom_eval_metrics[0],
        aggregate_ops=custom_eval_metric_aggregate_ops[0],
        version=0,
        dataset_name="HellaSwag",
    ),
    LeaderBoardTask(
        name=f"hellaswag_{custom_eval_tasktypes[1]}_10s",
        abbr="HellaSwag Option Key Circular(10 shot)",
        num_fewshot=10,
        use_cot=False,
        subtasks=[f"hellaswag_{custom_eval_tasktypes[1]}"],
        metrics=custom_eval_metrics[1],
        aggregate_ops=custom_eval_metric_aggregate_ops[1],
        version=0,
        dataset_name="HellaSwag",
    ),
    LeaderBoardTask(
        name=f"hellaswag_{custom_eval_tasktypes[2]}_10s",
        abbr="HellaSwag Greedy Option Key(10 shot)",
        num_fewshot=10,
        use_cot=False,
        subtasks=[f"hellaswag_{custom_eval_tasktypes[2]}"],
        metrics=custom_eval_metrics[2],
        aggregate_ops=custom_eval_metric_aggregate_ops[2],
        version=0,
        dataset_name="HellaSwag",
    ),
    LeaderBoardTask(
        name=f"hellaswag_{custom_eval_tasktypes[3]}_10s",
        abbr="HellaSwag Answer(10 shot)",
        num_fewshot=10,
        use_cot=False,
        subtasks=[f"hellaswag_{custom_eval_tasktypes[3]}"],
        metrics=custom_eval_metrics[3],
        aggregate_ops=custom_eval_metric_aggregate_ops[3],
        version=0,
        dataset_name="HellaSwag",
    ),
    # LeaderBoardTask(
    #     name="arc",
    #     abbr="ARC(25 shot)",
    #     num_fewshot=25,
    #     use_cot=False,
    #     subtasks=["arc_challenge"],
    #     metrics=["acc_norm"],
    #     aggregate_ops=['mean'],
    #     version=0,
    # ),
    # LeaderBoardTask
    # (
    #     name="agieval",
    #     abbr="AGIEval Eng QA(3-5 shot)",
    #     num_fewshot=5,
    #     use_cot=False,
    #     subtasks = [
    #         "agieval_eng_qa_lsat-ar",
    #         "agieval_eng_qa_lsat-lr",
    #         "agieval_eng_qa_lsat-rc",
    #         "agieval_eng_qa_logiqa-en",
    #         "agieval_eng_qa_sat-math",
    #         "agieval_eng_qa_sat-en",
    #         "agieval_eng_qa_aqua-rat",
    #         "agieval_eng_qa_sat-en-without-passage",
    #         "agieval_eng_qa_gaokao-english"
    #     ],
    #     metrics=["acc"],
    #     aggregate_ops=['mean'],
    #     version=0,
    # )
]

LEADERBOARDTASK_REGISTRY = \
{ item.name: item for item in leaderboard_tasks }

# agieval_qa = [
#     # 9 subtasks of qa
#     'agieval_eng_qa_lsat-ar',
#     'agieval_eng_qa_lsat-lr',
#     'agieval_eng_qa_lsat-rc',
#     'agieval_eng_qa_logiqa-en',
#     'agieval_eng_qa_sat-math',
#     'agieval_eng_qa_sat-en',
#     'agieval_eng_qa_aqua-rat',
#     'agieval_eng_qa_sat-en-without-passage',
#     'agieval_eng_qa_gaokao-english',
#     # 9 substask of cot qa
#     'agieval_eng_qa_cot_lsat-ar',
#     'agieval_eng_qa_cot_lsat-lr',
#     'agieval_eng_qa_cot_lsat-rc',
#     'agieval_eng_qa_cot_logiqa-en',
#     'agieval_eng_qa_cot_sat-math',
#     'agieval_eng_qa_cot_sat-en',
#     'agieval_eng_qa_cot_aqua-rat',
#     'agieval_eng_qa_cot_sat-en-without-passage',
#     'agieval_eng_qa_cot_gaokao-english',
#     # cloze
#     "agieval_eng_cloze",
#     # cloze cot
#     "agieval_eng_cloze_cot",
# ]
# models = [
#     'meta-llama/Llama-2-7b-hf',
#     'meta-llama/Llama-2-13b-hf',
#     'meta-llama/Llama-2-7b-chat-hf',
#     'meta-llama/Llama-2-13b-chat-hf',
#     'lmsys/vicuna-7b-v1.3',
# ]
# TASK = LEADERBOARDTASK_REGISTRY['mmlu_circular']
# NUM_FEWSHOT = TASK.num_fewshot
# MODEL_NAME = "distilgpt2"
# print(TASK)
# OUTPUT_DIR = f'/eval_results/{hash_md5(MODEL_NAME)}/{TASK.name}/{TASK.version}/{NUM_FEWSHOT}shot'
# os.makedirs(OUTPUT_DIR, exist_ok=True)
def encode_modelname(model_name):
    # can not
    return model_name.replace("/", ".")

def decoded_modelname(encoded_model_name):
    return encoded_model_name.replace(".", "/")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--leaderboard_task", required=True, choices=LEADERBOARDTASK_REGISTRY)
    parser.add_argument("--hf_model_name", type=str, required=True)
    parser.add_argument("--batch_size", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--limit", type=float, default=None,
                        help="Limit the number of examples per task. "
                             "If <1, limit is a percentage of the total number of examples.")
    parser.add_argument("--num_fewshot", type=int, default=None)
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--use_data_parallel", action="store_true")
    parser.add_argument("--use_model_parallel", action="store_true")
    parser.add_argument("--output_base_path", type=str, default="/eval_results")
    return parser.parse_args()

def eval_and_dump(leaderboardtask_name, hf_model_name, batch_size, device, no_cache, limit, use_data_parallel, use_model_parallel, num_fewshot, output_base_path):
    # setting leaderboard task eval settings
    task=LEADERBOARDTASK_REGISTRY[leaderboardtask_name]
    print(task)
    if num_fewshot is not None:
        print(f'WARNING: OVERWRITING NUM_FEWSHOT BY COMMAND ARGUMENT "--num_fewshot {num_fewshot}" INSTEAD of TASK DEFAULT NUM_FEWSHOT {task.num_fewshot}')
    else:
        num_fewshot=task.num_fewshot

    output_dir=f'{output_base_path}/{encode_modelname(hf_model_name)}/{task.name}/{task.version}/{num_fewshot}shot'
    os.makedirs(output_dir, exist_ok=True)
    output_path=f"{output_dir}/results.json"

    # check task names(a leader board task could have a set of tasks, e.g. MMLU)
    task_names=utils.pattern_match(task.subtasks, tasks.ALL_TASKS)
    print(f"Selected Tasks: {task_names}")
    
    model='hf-causal-experimental'
    model_args=f"pretrained={hf_model_name}"
    if use_model_parallel:
        model_args+=",use_accelerate=True"
        device="auto"

    print(f"[INFO] num_fewshot: {num_fewshot}")
    results = evaluator.simple_evaluate(
        model=model,
        model_args=model_args,
        tasks=task_names,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        device=device,
        no_cache=no_cache,
        limit=limit,
        write_out=True,
        output_base_path=output_dir,
        use_data_parallel_accelerate=use_data_parallel,
    )

    # if using data parallel, eval on multiple process, let the first process dump result json file to avoid writing conflict.
    if not use_data_parallel or (use_data_parallel and results["is_main_process"]):
        dumped = json.dumps(results, indent=2)
        print(dumped)
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w") as f:
                print(f"[INFO] writed result to {output_path}")
                f.write(dumped)

        batch_sizes = ",".join(map(str, results["config"]["batch_sizes"]))
        print(
            f"{model} ({model_args}), limit: {limit}, provide_description: {None}, "
            f"num_fewshot: {num_fewshot}, batch_size: {batch_size}{f' ({batch_sizes})' if batch_sizes else ''}"
        )
        print(evaluator.make_table(results))

def main():
    args = parse_args()
    
    # check data parallel arguments
    current_env = os.environ.copy()
    if args.use_data_parallel:
        if not "MASTER_PORT" in current_env:
            raise("when using data_parallel, the eval script be should launch with accelerate launch ...")
        
    if "MASTER_PORT" in current_env and not args.use_data_parallel:
        print("WARNING: --use_data_parallel should be used if you are using accelerate to eval in data parallel, otherwise, the evaluation and result dumping will not work normally")

    if args.limit:
        print(
            "WARNING: --limit SHOULD ONLY BE USED FOR TESTING. REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT."
        )

    if args.use_data_parallel and args.use_model_parallel:
        raise(
            "not support use model parallel and data parallel together."
        )
    
    print(f"Selected Leader Board Task: {args.leaderboard_task}")
    
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

if __name__ == "__main__":
    main()

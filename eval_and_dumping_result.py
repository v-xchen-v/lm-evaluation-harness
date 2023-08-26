import argparse
import json
import logging
import os

from lm_eval import tasks, evaluator, utils
from lm_eval.tasks.hendrycks_test import SUBJECTS as MMLU_SUBJECTS

logging.getLogger("openai").setLevel(logging.WARNING)

def hash_md5(model_name: str):
    import hashlib
    return hashlib.md5(model_name.encode('utf-8')).hexdigest()

from dataclasses import dataclass

@dataclass
class LeaderBoardTask:
    # task name, which is also the directory name contains task result files
    name : str

    # the abbr name of task shows on the leadleader board table header
    abbr : str

    # # the model to apply evaluation task on
    # model_name: str

    # the fewshot number
    num_fewshot: int

    # using cot or not, for AGIEval benchmark
    use_cot: bool

    # the subtasks included
    subtasks : list[str]

    # the selected single metric showed in leaderboard table
    metric: str

    # appragate subtasks, support 'mean' only for now
    aggregate_op: str

    version: int

# TODO: as config file
leaderboard_tasks = [
    LeaderBoardTask(
        name="mmlu",
        abbr="MMLU(5 shot)",
        num_fewshot=5,
        use_cot=False,
        subtasks= [f'hendrycksTest-{sub}' for sub in  MMLU_SUBJECTS],
        metric="acc",
        aggregate_op='mean',
        version=1,
    ),
    LeaderBoardTask(
        name="mmlu_circular",
        abbr="MMLU CircularChoices(5 shot)",
        num_fewshot=5,
        use_cot=False,
        subtasks= [f"hendrycksTest-CircularChoices-{sub}" for sub in MMLU_SUBJECTS],
        metric="acc_circularchoices",
        aggregate_op='mean',
        version=0,
    ),
    LeaderBoardTask(
        name="truthfulqa",
        abbr="TruthfulQA(0 shot)",
        num_fewshot=0,
        use_cot = False,
        subtasks=["truthfulqa_mc"],
        metric = "mc2",
        aggregate_op='mean',
        version=1
    ),
    LeaderBoardTask(
        name="hellaswag",
        abbr="HellaSwag(10 shot)",
        num_fewshot=10,
        use_cot=False,
        subtasks=["hellaswag"],
        metric="acc_norm",
        aggregate_op='mean',
        version=0,
    ),
    LeaderBoardTask(
        name="arc",
        abbr="ARC(25 shot)",
        num_fewshot=25,
        use_cot=False,
        subtasks=["arc_challenge"],
        metric="acc_norm",
        aggregate_op='mean',
        version=0,
    ),
    LeaderBoardTask
    (
        name="agieval",
        abbr="AGIEval Eng QA(3-5 shot)",
        num_fewshot=5,
        use_cot=False,
        subtasks = [
            "agieval_eng_qa_lsat-ar",
            "agieval_eng_qa_lsat-lr",
            "agieval_eng_qa_lsat-rc",
            "agieval_eng_qa_logiqa-en",
            "agieval_eng_qa_sat-math",
            "agieval_eng_qa_sat-en",
            "agieval_eng_qa_aqua-rat",
            "agieval_eng_qa_sat-en-without-passage",
            "agieval_eng_qa_gaokao-english"
        ],
        metric="acc",
        aggregate_op='mean',
        version=0,
    )
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
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--use_data_parallel", action="store_true")
    parser.add_argument("--use_model_parallel", action="store_true")
    return parser.parse_args()

def eval_and_dump(leaderboardtask_name, hf_model_name, batch_size, device, no_cache, limit, use_data_parallel, use_model_parallel):
    # setting leaderboard task eval settings
    task=LEADERBOARDTASK_REGISTRY[leaderboardtask_name]
    print(task)
    num_fewshot=task.num_fewshot
    output_dir=f'/eval_results/{encode_modelname(hf_model_name)}/{task.name}/{task.version}/{num_fewshot}shot'
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
    )

    dumped = json.dumps(results, indent=2)
    print(dumped)

    if output_path:
        # if using data parallel, eval on multiple process, let the first process dump result json file to avoid writing conflict.
        if not use_data_parallel or (use_data_parallel and results["distributed_process_id"] == 0):
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w") as f:
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
    )

if __name__ == "__main__":
    main()

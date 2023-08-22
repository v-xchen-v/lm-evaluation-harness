import argparse
import json
import logging
import os

from lm_eval import tasks, evaluator, utils

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

leaderboard_tasks = [
    LeaderBoardTask(
        name="mmlu",
        abbr="MMLU(5 shot)",
        num_fewshot=5,
        use_cot=False,
        subtasks=[
            "hendrycksTest-abstract_algebra",
            "hendrycksTest-anatomy",
            "hendrycksTest-astronomy",
            "hendrycksTest-business_ethics",
            "hendrycksTest-clinical_knowledge",
            "hendrycksTest-college_biology",
            "hendrycksTest-college_chemistry",
            "hendrycksTest-college_computer_science",
            "hendrycksTest-college_mathematics",
            "hendrycksTest-college_medicine",
            "hendrycksTest-college_physics",
            "hendrycksTest-computer_security",
            "hendrycksTest-conceptual_physics",
            "hendrycksTest-econometrics",
            "hendrycksTest-electrical_engineering",
            "hendrycksTest-elementary_mathematics",
            "hendrycksTest-formal_logic",
            "hendrycksTest-global_facts",
            "hendrycksTest-high_school_biology",
            "hendrycksTest-high_school_chemistry",
            "hendrycksTest-high_school_computer_science",
            "hendrycksTest-high_school_european_history",
            "hendrycksTest-high_school_geography",
            "hendrycksTest-high_school_government_and_politics",
            "hendrycksTest-high_school_macroeconomics",
            "hendrycksTest-high_school_mathematics",
            "hendrycksTest-high_school_microeconomics",
            "hendrycksTest-high_school_physics",
            "hendrycksTest-high_school_psychology",
            "hendrycksTest-high_school_statistics",
            "hendrycksTest-high_school_us_history",
            "hendrycksTest-high_school_world_history",
            "hendrycksTest-human_aging",
            "hendrycksTest-human_sexuality",
            "hendrycksTest-international_law",
            "hendrycksTest-jurisprudence",
            "hendrycksTest-logical_fallacies",
            "hendrycksTest-machine_learning",
            "hendrycksTest-management",
            "hendrycksTest-marketing",
            "hendrycksTest-medical_genetics",
            "hendrycksTest-miscellaneous",
            "hendrycksTest-moral_disputes",
            "hendrycksTest-moral_scenarios",
            "hendrycksTest-nutrition",
            "hendrycksTest-philosophy",
            "hendrycksTest-prehistory",
            "hendrycksTest-professional_accounting",
            "hendrycksTest-professional_law",
            "hendrycksTest-professional_medicine",
            "hendrycksTest-professional_psychology",
            "hendrycksTest-public_relations",
            "hendrycksTest-security_studies",
            "hendrycksTest-sociology",
            "hendrycksTest-us_foreign_policy",
            "hendrycksTest-virology",
            "hendrycksTest-world_religions"
        ],
        metric="acc",
        aggregate_op='mean',
        version=1,
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

# mmlu = [ 
#         "hendrycksTest-abstract_algebra",
#         "hendrycksTest-anatomy",
#         "hendrycksTest-astronomy",
#         "hendrycksTest-business_ethics",
#         "hendrycksTest-clinical_knowledge",
#         "hendrycksTest-college_biology",
#         "hendrycksTest-college_chemistry",
#         "hendrycksTest-college_computer_science",
#         "hendrycksTest-college_mathematics",
#         "hendrycksTest-college_medicine",
#         "hendrycksTest-college_physics",
#         "hendrycksTest-computer_security",
#         "hendrycksTest-conceptual_physics",
#         "hendrycksTest-econometrics",
#         "hendrycksTest-electrical_engineering",
#         "hendrycksTest-elementary_mathematics",
#         "hendrycksTest-formal_logic",
#         "hendrycksTest-global_facts",
#         "hendrycksTest-high_school_biology",
#         "hendrycksTest-high_school_chemistry",
#         "hendrycksTest-high_school_computer_science",
#         "hendrycksTest-high_school_european_history",
#         "hendrycksTest-high_school_geography",
#         "hendrycksTest-high_school_government_and_politics",
#         "hendrycksTest-high_school_macroeconomics",
#         "hendrycksTest-high_school_mathematics",
#         "hendrycksTest-high_school_microeconomics",
#         "hendrycksTest-high_school_physics",
#         "hendrycksTest-high_school_psychology",
#         "hendrycksTest-high_school_statistics",
#         "hendrycksTest-high_school_us_history",
#         "hendrycksTest-high_school_world_history",
#         "hendrycksTest-human_aging",
#         "hendrycksTest-human_sexuality",
#         "hendrycksTest-international_law",
#         "hendrycksTest-jurisprudence",
#         "hendrycksTest-logical_fallacies",
#         "hendrycksTest-machine_learning",
#         "hendrycksTest-management",
#         "hendrycksTest-marketing",
#         "hendrycksTest-medical_genetics",
#         "hendrycksTest-miscellaneous",
#         "hendrycksTest-moral_disputes",
#         "hendrycksTest-moral_scenarios",
#         "hendrycksTest-nutrition",
#         "hendrycksTest-philosophy",
#         "hendrycksTest-prehistory",
#         "hendrycksTest-professional_accounting",
#         "hendrycksTest-professional_law",
#         "hendrycksTest-professional_medicine",
#         "hendrycksTest-professional_psychology",
#         "hendrycksTest-public_relations",
#         "hendrycksTest-security_studies",
#         "hendrycksTest-sociology",
#         "hendrycksTest-us_foreign_policy",
#         "hendrycksTest-virology",
#         "hendrycksTest-world_religions"
#     ]

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
models = [
    'meta-llama/Llama-2-7b-hf',
    'meta-llama/Llama-2-13b-hf',
    'meta-llama/Llama-2-7b-chat-hf',
    'meta-llama/Llama-2-13b-chat-hf',
    'lmsys/vicuna-7b-v1.3',
]
TASK = LEADERBOARDTASK_REGISTRY['agieval']
NUM_FEWSHOT = TASK.num_fewshot_
MODEL_NAME = "lmsys/vicuna-7b-v1.3"
print(TASK)
OUTPUT_DIR = f'/eval_results/{hash_md5(MODEL_NAME)}/{TASK.name}/{TASK.version}/{NUM_FEWSHOT}shot'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='hf-causal-experimental')
    parser.add_argument("--model_args", default=f"pretrained={MODEL_NAME}")
    # parser.add_argument("--tasks", default='agieval_eng_qa_lsat-ar,agieval_eng_qa_lsat-lr,agieval_eng_qa_lsat-rc,agieval_eng_qa_logiqa-en,agieval_eng_qa_sat-math,agieval_eng_qa_sat-en,agieval_eng_qa_aqua-rat,agieval_eng_qa_sat-en-without-passage,agieval_eng_qa_gaokao-english', choices=utils.MultiChoice(tasks.ALL_TASKS))
    parser.add_argument("--tasks", default=','.join(TASK.subtasks), choices=utils.MultiChoice(tasks.ALL_TASKS))
    parser.add_argument("--provide_description", action="store_true")
    parser.add_argument("--num_fewshot", type=int, default=NUM_FEWSHOT)
    parser.add_argument("--batch_size", type=str, default=None)
    parser.add_argument("--max_batch_size", type=int, default=None,
                        help="Maximal batch size to try with --batch_size auto")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_path", default=f"{OUTPUT_DIR}/results.json")
    parser.add_argument("--limit", type=float, default=None,
                        help="Limit the number of examples per task. "
                             "If <1, limit is a percentage of the total number of examples.")
    parser.add_argument("--data_sampling", type=float, default=None)
    parser.add_argument("--no_cache", action="store_true", default=
    True)
    parser.add_argument("--decontamination_ngrams_path", default=None)
    parser.add_argument("--description_dict_path", default=None)
    parser.add_argument("--check_integrity", action="store_true")
    parser.add_argument("--write_out", action="store_true", default=True)
    parser.add_argument("--output_base_path", type=str, default=OUTPUT_DIR)

    return parser.parse_args()


def main():
    args = parse_args()

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

    if args.output_path:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        with open(args.output_path, "w") as f:
            f.write(dumped)

    batch_sizes = ",".join(map(str, results["config"]["batch_sizes"]))
    print(
        f"{args.model} ({args.model_args}), limit: {args.limit}, provide_description: {args.provide_description}, "
        f"num_fewshot: {args.num_fewshot}, batch_size: {args.batch_size}{f' ({batch_sizes})' if batch_sizes else ''}"
    )
    print(evaluator.make_table(results))


if __name__ == "__main__":
    main()

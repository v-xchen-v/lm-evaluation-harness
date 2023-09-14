import argparse
import json
import logging
import os

from lm_eval import tasks, evaluator, utils

logging.getLogger("openai").setLevel(logging.WARNING)


# task_name="hendrycksTest-abstract_algebra_greedyoptionkey"
# task_name="hendrycksTest-abstract_algebra_greedyanswer"
# task_name="hendrycksTest-econometrics_likelihoodoptionkeycircular"
# task_name="hendrycksTest-abstract_algebra_likelihoodoptioncontent"
# task_name="hellaswag_greedyoptionkey"
# task_name="hellaswag_greedyanswer"
# task_name="hellaswag_likelihoodoptionkeycircular"
task_name="hellaswag_likelihoodoptioncontent"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='hf-causal-experimental')
    # parser.add_argument("--model_args", default="pretrained=meta-llama/Llama-2-7b-chat-hf,use_accelerate=True")
    parser.add_argument("--model_args", default="pretrained=distilgpt2")
    # parser.add_argument("--tasks", default="hellaswag", choices=utils.MultiChoice(tasks.ALL_TASKS))
    # parser.add_argument("--tasks", default=f"{task_name}", choices=utils.MultiChoice(tasks.ALL_TASKS))
    parser.add_argument("--tasks", default=f"{task_name}", choices=utils.MultiChoice(tasks.ALL_TASKS))
    parser.add_argument("--provide_description", action="store_true")
    parser.add_argument("--num_fewshot", type=int, default=10)
    parser.add_argument("--batch_size", type=str, default=None)
    parser.add_argument("--max_batch_size", type=int, default=None,
                        help="Maximal batch size to try with --batch_size auto")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_path", default=f"./output/{task_name}_results.json")
    parser.add_argument("--limit", type=float, default=2,
                        help="Limit the number of examples per task. "
                             "If <1, limit is a percentage of the total number of examples.")
    parser.add_argument("--data_sampling", type=float, default=None)
    parser.add_argument("--no_cache", action="store_true", default=True)
    parser.add_argument("--decontamination_ngrams_path", default=None)
    parser.add_argument("--description_dict_path", default=None)
    parser.add_argument("--check_integrity", action="store_true")
    parser.add_argument("--write_out", action="store_true", default=True)
    parser.add_argument("--output_base_path", type=str, default="./output")

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

    args.output_path = f"./output/{args.tasks}_results.json"
    
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

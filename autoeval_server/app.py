"""
1. show eval results
2. submit eval tasks
"""

import gradio as gr
from eval_and_dumping_result import LEADERBOARDTASK_REGISTRY, leaderboard_tasks
from autoeval_server.resultparser.loadresults import EvalTaskResultInfo
from config import RESULTS_SAVE_ROOT
from autoeval_server.evalrunner.evaltask import EvalModelTask
from autoeval_server.evalrunner.taskqueue import task_queue
from autoeval_server.evalrunner.taskstatus import get_finished_evaluations, get_pending_evaluations, get_running_evaluations
from typing import List
# import gradio as gr
import pandas as pd
from autoeval_server.leaderboard.display import get_leaderboard_df_data
from autoeval_server.leaderboard.display import DISPLAY_DATASETS

LEADERBOARDTASK_ABBR_REGISTRY = \
{ item.abbr: item for item in leaderboard_tasks }
LEADERBOARDTASK_ABBR = [
    item.abbr for item in leaderboard_tasks
]

def submit_tasks(model_name, selection_tasks, overwrite_if_exists):
    # print(model_name)
    # print(selection_tasks)
    for selected_task in [LEADERBOARDTASK_ABBR_REGISTRY[x] for x in selection_tasks]:
        if EvalTaskResultInfo.from_evaltask(results_save_root=RESULTS_SAVE_ROOT, model_id=model_name, leaderboardtask=selected_task).is_result_exists() and not overwrite_if_exists:
            continue
        task_queue.add_task(EvalModelTask(model=model_name, eval_task=selected_task))
    refresh_evaluation_status_tb()

def get_leaderboard_df(dataset_name: str, num_fewshot: int, use_cot: bool):
    # example = pd.DataFrame.from_records([
    #     {
    #        "Model name": "huggyllama/llama-7b",
    #        "MMLU": 0,
    #        "TruthfulQA(0-s)": 0, 
    #        "Hellaswag(10-s)": 0, 
    #        "ARC(25-s)": 0, 
    #        "AGIEval(0-s)": 0, 
    #        "AGIEval(5-s)": 0, 
    #        "AGIEval(0-s CoT)":0
    #     }
    # ]
    # )
    # return example

    df = pd.DataFrame.from_records(get_leaderboard_df_data(dataset_name, num_fewshot, use_cot))
    return df

# example_finished_tasks =
# [
#     ["huggyllama/llama-7b", "MMLU"], 
#     ["huggyllama/llama-7b", "Hellaswag"],
#     ["huggyllama/llama-7b", "TruthfulQA"],
#     ["huggyllama/llama-7b", "ARC"],
#     ["huggyllama/llama-7b", "AGIEval"],
# ]

# example_pending_tasks = \
# [
#     ["huggyllama/llama-13b", "MMLU"],
# ]

# example_processing_task = \
# [
#     ["huggyllama/llama-13b", "Hellaswag"],
#     ["huggyllama/llama-13b", "TruthfulQA"],
#     ["huggyllama/llama-13b", "ARC"],
#     ["huggyllama/llama-13b", "AGIEval"],
#     ["huggyllama/llama-30b", "MMLU"],
#     ["huggyllama/llama-30b", "Hellaswag"],
#     ["huggyllama/llama-30b", "TruthfulQA"],
#     ["huggyllama/llama-30b", "ARC"],
#     ["lmsys/vicuna-7b-v1.1", "AGIEval"],
#     ["lmsys/vicuna-7b-v1.1", "Hellaswag"],
#     ["lmsys/vicuna-7b-v1.1", "TruthfulQA"],
#     ["lmsys/vicuna-7b-v1.1", "ARC"],
#     ["lmsys/vicuna-7b-v1.1", "AGIEval"],
# ]

def parse_evaluation_table(tasks: List[EvalModelTask]):
    if len(tasks) > 0:
        # print(tasks)
        return [[t.model, t.eval_task.abbr] for t in tasks]
    else:
        # should have a least one element, otherwise web UI will hang (I think it's a bug of gradio)
        return [['-'], ['-']]

finished_tasks = []
pending_tasks = []
processing_task = []


finished_tasks = parse_evaluation_table(get_finished_evaluations())
pending_tasks = parse_evaluation_table(get_pending_evaluations())
processing_task = parse_evaluation_table(get_running_evaluations())

display_dataset_dict = {}

def refresh_evaluation_status_tb():
    print("[INFO] refreshing evaluation tables")
    finished_tasks = parse_evaluation_table(get_finished_evaluations())
    pending_tasks = parse_evaluation_table(get_pending_evaluations())
    processing_task = parse_evaluation_table(get_running_evaluations())
    leaderboard_dfs = [get_leaderboard_df(dataset.dataset_name, dataset.num_fewshot, dataset.use_cot) for dataset in DISPLAY_DATASETS]
    print("[INFO] refreshed")
    return [finished_tasks, pending_tasks, processing_task, *leaderboard_dfs]



with gr.Blocks() as demo:
    with gr.Tabs(elem_classes="tab-buttons") as tabs:
        with gr.TabItem("🏅 LLM Benchmark (lite)", elem_id="llm-benchmark-tab-table", id=0):
            gr.Markdown("Leader Board")
            for dataset in DISPLAY_DATASETS:
                with gr.TabItem(dataset.abbr):
                    display_dataset_dict[dataset.abbr] = gr.Dataframe(
                    # leaderboard_table = gr.Dataframe(
                        # headers=["Model Name", "MMLU(5-s)", "TruthfulQA(0-s)", "Hellaswag(10-s)", "ARC(25-s)", "AGIEval(0-s)", "AGIEval(5-s)", "AGIEval(0-s CoT)", "AGIEval(5-s CoT)"],
                        datatype=["markdown", "number", "number", "number", "number", "number"],
                        interactive=False,
                        value=get_leaderboard_df(dataset.dataset_name, dataset.num_fewshot, dataset.use_cot),
                        # set wrap to true, so that the many columns can show in one page
                        # col_count=(6, "fixed"),
                        wrap=True,
                    )
        with gr.TabItem("✉️✨ Submit here! ", elem_id="llm-benchmark-tab-table", id=1):            
            gr.Markdown("✉️✨ Submit your model here!")
            # submit new task section
            gr.Markdown("These models will be automatically evaluated on server")
            model_name = gr.Textbox(label="Model name", placeholder="What is your model name")
            tasks_selection = gr.Dropdown(choices=LEADERBOARDTASK_ABBR, value=[LEADERBOARDTASK_ABBR[0]], multiselect=True, label="benchmarks", info="Select the eval tasks.", interactive=True)
            with gr.Row():
                submit_button = gr.Button('Submit Eval')
                overwrite_result=gr.Checkbox(label="Overwrite If Exists")
            
            out = None
            
            # submit a task
            submit_button.click(fn=submit_tasks, inputs=[model_name, tasks_selection, overwrite_result], outputs=None)

            # # input fewshot number and with/without CoT
            # with gr.Accordion("Evaluation settings", open=False):
            #     gr.Slider(0, 25, value=0, label="n-shot", info="Choose fewshot number, default as MMLU(5-s), TruthfulQA(0-s), ARC(25-s), HellaSwag(10-s), AGIEval(5-s)", step=1)
            #     gr.Checkbox(value=False, label="use CoT or not", info="only AGIEval support CoT", interactive=True)

            # list finished tasks section
            with gr.Accordion("✅Finished Evaluations", open=False):
                finished_eval_table = gr.components.Dataframe(
                    value=finished_tasks,
                    headers=["Model name", "Task name"],
                    datatype=["str", "str"])
                
            # list pending tasks section
            with gr.Accordion("⏳Pending Evaluations", open=False):
                pending_eval_table = gr.components.Dataframe(
                    value=pending_tasks,
                    headers=["Model name", "Task name"],
                    datatype=["str", "str"])
                
            # list  tasks section
            with gr.Accordion("🔄Running Evaluations", open=False):
                running_eval_table = gr.components.Dataframe(
                    value=processing_task,
                    headers=["Model name", "Task name"],
                    datatype=["str", "str"])
            
            refresh_btn = gr.Button("Refresh")
            refresh_btn.click(
                fn=refresh_evaluation_status_tb,
                inputs=[],
                outputs=[
                    finished_eval_table,
                    pending_eval_table,
                    running_eval_table,
                    *list(display_dataset_dict.values())
                ]
            )

    # refresh the evalution status table every 3 minite.
    REFRESH_INTERNAL = 60*3
    demo.load(
                fn=refresh_evaluation_status_tb,
                inputs=[],
                outputs=[
                    finished_eval_table,
                    pending_eval_table,
                    running_eval_table,
                    *list(display_dataset_dict.values())
                ],
                every=REFRESH_INTERNAL
            )
    
if __name__ == "__main__":
    demo.queue().launch(debug=True)
    # demo.queue().launch(share=True)
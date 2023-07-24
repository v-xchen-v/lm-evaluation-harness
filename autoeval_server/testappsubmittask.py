import gradio as gr
from config import LEADERBOARDTASKS, LEADERBOARDTASK_REGISTRY
from evalrunner.evaltask import EvalModelTask
from evalrunner.taskqueue import task_queue
from evalrunner.taskstatus import get_finished_evaluations, get_pending_evaluations, get_running_evaluations
from resultparser.loadresults import is_result_exists
from typing import List

def submit_tasks(model_name, selection_tasks, overwrite_if_exists):
    # print(model_name)
    # print(selection_tasks)
    for selected_task in [LEADERBOARDTASK_REGISTRY[x] for x in selection_tasks]:
        if is_result_exists(model_name=model_name, task_name=selected_task.name, version=selected_task.task_version, num_fewshot=selected_task.num_fewshot) and not overwrite_if_exists:
            continue
        task_queue.add_task(EvalModelTask(model=model_name, eval_task=selected_task))
    refresh_evaluation_status_tb()

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
        return []

finished_tasks = []
pending_tasks = []
processing_task = []


finished_tasks = parse_evaluation_table(get_finished_evaluations())
pending_tasks = parse_evaluation_table(get_pending_evaluations())
processing_task = parse_evaluation_table(get_running_evaluations())

def refresh_evaluation_status_tb():
    print("[INFO] refreshing evaluation tables")
    finished_tasks = parse_evaluation_table(get_finished_evaluations())
    pending_tasks = parse_evaluation_table(get_pending_evaluations())
    processing_task = parse_evaluation_table(get_running_evaluations())
    return [finished_tasks, pending_tasks, processing_task]

with gr.Blocks() as demo:
    gr.Markdown("✉️✨ Submit your model here!")
    # submit new task section
    gr.Markdown("These models will be automatically evaluated on server")
    model_name = gr.Textbox(label="Model name", placeholder="What is your model name")
    tasks_selection = gr.Dropdown(choices=LEADERBOARDTASKS, value=[LEADERBOARDTASKS[0]], multiselect=True, label="benchmarks", info="Select the eval tasks.", interactive=True)
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
            datatype=["str", "str"], 
            max_rows=5)
        
    # list pending tasks section
    with gr.Accordion("⏳Pending Evaluations", open=False):
        pending_eval_table = gr.components.Dataframe(
            value=pending_tasks,
            headers=["Model name", "Task name"],
            datatype=["str", "str"], 
            max_rows=5)
        
    # list  tasks section
    with gr.Accordion("🔄Running Evaluations", open=False):
        running_eval_table = gr.components.Dataframe(
            value=processing_task,
            headers=["Model name", "Task name"],
            datatype=["str", "str"],  
            max_rows=5)
    
    refresh_btn = gr.Button("Refresh")
    refresh_btn.click(
        refresh_evaluation_status_tb,
        inputs=[],
        outputs=[
            finished_eval_table,
            pending_eval_table,
            running_eval_table
        ]    )

    # refresh the evalution status table every 1 minite.
    demo.load(fn=refresh_evaluation_status_tb, inputs=[], outputs=[
            finished_eval_table,
            pending_eval_table,
            running_eval_table
        ], every=10*1)
    
if __name__ == "__main__":
    demo.queue().launch()
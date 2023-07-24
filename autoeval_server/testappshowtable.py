import gradio as gr
import pandas as pd
from leaderboard.display import get_leaderboard_df_data

def get_leaderboard_df():
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

    df = pd.DataFrame.from_records(get_leaderboard_df_data())
    return df

# def refresh():
#     leaderboard_df = get_leaderboard_df()

with gr.Blocks() as demo:
    gr.Markdown("Leader Board")
    leaderboard_table = gr.Dataframe(
        headers=["Model Name", "MMLU(5-s)", "TruthfulQA(0-s)", "Hellaswag(10-s)", "ARC(25-s)", "AGIEval(0-s)", "AGIEval(5-s)", "AGIEval(0-s CoT)", "AGIEval(5-s CoT)"],
        datatype=["markdown", "number", "number", "number", "number", "number", "number", "number", "number"],
        interactive=False,
        value=get_leaderboard_df(),
        # set wrap to true, so that the many columns can show in one page
        wrap=True
    )
    # refresh_btn = gr.Button("Refresh")
    # refresh_btn.click(
    #     refresh,
    #     inputs=[],
    #     outputs=[
    #         leaderboard_table
    #     ]
    # )

if __name__ == "__main__":
    demo.launch()
from dataclasses import dataclass

@dataclass
class LeaderBoardTask:
    # task name, which is also the directory name contains task result files
    name : str

    # the abbr name of task shows on the leadleader board table header
    abbr : str

    # the version of tasks, start from 0
    task_version: int

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



if __name__ == "__main__":
    example = LeaderBoardTask(
        name="agieval",
        abbr="agieval_eng_qa(3-5s)", 
        task_version=0,
        num_fewshot=5,
        use_cot=False,
        subtasks=[
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
        aggregate_op='mean'
    )
    print(example)
    print(LEADERBOARDTASK_REGISTRY)
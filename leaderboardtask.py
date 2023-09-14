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

    # the selected metrics showed in leaderboard table
    metrics: list[str]
    # metric: str

    # appragate subtasks, support 'mean' only for now
    aggregate_ops: str

    version: int
    
    # task dataset is belong to, which is also the key to group tasks
    dataset_name: str
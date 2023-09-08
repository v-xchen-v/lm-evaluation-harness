from leaderboard.leaderboardtasks import LeaderBoardTask
from dataclasses import dataclass

@dataclass
class EvalModelTask:
    model: str
    eval_task: LeaderBoardTask
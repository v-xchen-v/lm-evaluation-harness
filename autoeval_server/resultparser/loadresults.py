import json
from autoeval_server.resultparser.getmodelinfo import ModelInfo
from typing import Tuple
from config import RESULTS_SAVE_ROOT
from eval_and_dumping_result import LEADERBOARDTASK_REGISTRY
from collections import defaultdict
# from dataclasses import dataclass
# # load eval result to memory dictionary
from pathlib import Path
import numpy as np
import os
from collections import ChainMap
import pickle
from leaderboardtask import LeaderBoardTask
# list_models, is_result_exists, get_leaderboard_aggregated_metrics
import itertools

num_of_doc_cache_filepath = '.num_of_doc.pkl'

def _list_subdirectories_by_level(directory, slevel=1):
    """
    list the subdirectories by level.
    """

    if not Path(directory).is_dir():
        raise Exception(f"The directory not exists")
    
    if slevel < 1:
         raise "slevel should bigger than 1"
    subdirectories = []
    stack = [(directory, 0)]

    while True:
        current_dir, level = stack.pop()
        if level == slevel-1:
            for item_path in Path(current_dir).iterdir():
                  if Path(item_path).is_dir():
                    subdirectories.append(item_path)
            break
        
        if level < slevel-1:
            for item_path in Path(current_dir).iterdir():
                    stack.append((item_path, level + 1))

    return subdirectories

def _contains_results_json_file(dir, result_file_pattern ="results.json"):
    return len(list(Path(dir).glob(f'**/{result_file_pattern}'))) > 0

def list_models():
    """
    list model have results
    """
    model_dirs = [Path(dirname) for dirname in _list_subdirectories_by_level(RESULTS_SAVE_ROOT, 1)]
    
    # have_results_model_dirs = [x for x in model_dirs if _contains_results_json_file(x)]
    
    model_ids = [ModelInfo.get_decoded_modelname(Path(dirname).name) for dirname in model_dirs]
    return model_ids
    
# def list_tasks(model_name):
#     return [Path(dirname).name for dirname in list_subdirectories_by_level(Path(results_save_root)/ModelInfo.get_decoded_modelname(model_name), 1)]

# def list_tasks_version(model_name, tasks_name):
#     return [Path(dirname).name for dirname in list_subdirectories_by_level(Path (results_save_root)/ModelInfo.get_decoded_modelname(model_name)/tasks_name, 1)]



# def get_lastest_tasks_version(model_id, tasks_name):
#     task_dir = Path (results_save_root)/ModelInfo.encode_modelname(model_id)/tasks_name
#     if Path(task_dir).is_dir() and len(_list_subdirectories_by_level(task_dir, 1))>=1:
#         return max(map(int, [Path(dirname).name for dirname in _list_subdirectories_by_level(task_dir, 1)]))
#     else:
#         return None

# def list_subtasks_result_jsons(model_name, tasks_name, num_fewshot):
#     dirname = Path(results_save_root)/ModelInfo.get_decoded_modelname(model_name)/tasks_name
#     tasks_version = get_lastest_tasks_version(model_name, tasks_name)
#     if tasks_version is not None:
#         dirname = dirname/str(tasks_version)/f"{num_fewshot}shot"
#         return list(dirname.glob(f"results.json"))
#     return None

# def get_subtask_latestversion_result_jsonpath(model_name, task_name, num_fewshot):
#     dirname = Path(results_save_root)/ModelInfo.get_decoded_modelname(model_name)/task_name
#     tasks_version = get_lastest_tasks_version(model_name, task_name)
#     jsonpath = dirname/str(tasks_version)/f"{num_fewshot}shot"/'results.json'
#     return jsonpath

# def is_lastestversion_result_exists(model_name, task_name, num_fewshot):
#     return Path(get_subtask_latestversion_result_jsonpath(model_name, task_name, num_fewshot)).is_file()

# def get_subtask_result_jsonpath(model_name, task_name, version, num_fewshot):
#     dirname = Path(results_save_root)/ModelInfo.get_decoded_modelname(model_name)/task_name
#     jsonpath = dirname/str(version)/f"{num_fewshot}shot"/'results.json'
#     return jsonpath

# def is_result_exists(model_name, task_name, version, num_fewshot):
#     return Path(get_subtask_result_jsonpath(model_name, task_name, version,num_fewshot)).is_file()

# def parse_tasks_eval_results(json_filepath: str):
#     """
#     parse result per task
#     returns: a dict contains subtasks result
#     """
#     with open(json_filepath, 'r') as f:
#         # dict of subtask name to dict of metricname to value
#         results = json.load(f)['results']

#     return results

# def get_metric(model_id, task_name, num_fewshot, use_cot, subtask_name, metric_name):
#     hashed_modelname = ModelInfo.encode_modelname(model_id)
#     task_save_root = os.path.join(results_save_root,hashed_modelname,task_name)
#     result_jsonpath = os.path.join(task_save_root,str(get_lastest_tasks_version(model_id, task_name)),f'{num_fewshot}shot', 'results.json')
#     return parse_tasks_eval_results(result_jsonpath)[subtask_name][metric_name]

# def get_leaderboard_metrics(model_name, leaderboard_task):
#     """returns a diction of metric name to a list of metric value belongs to task subjects
#     example:
#         {'ppl_argmax_acc': [0.22, 0.3111111111111111, 0.375, 0.57, 0.3169811320754717, 0.3263888888888889, 0.27, 0.2, 0.13, 0.28901734104046245, 0.2549019607843137, 0.4, 0.44680851063829785, 0.21929824561403508, 0.2827586206896552, 0.37566137566137564, 0.35714285714285715, 0.36, 0.2967741935483871, 0.21182266009852216, 0.35, 0.3212121212121212, 0.3282828282828283, 0.47150259067357514, 0.3435897435897436, 0.17037037037037037, 0.38235294117647056, 0.26490066225165565, 0.42018348623853213, 0.28703703703703703, 0.37745098039215685, 0.350210970464135, 0.4349775784753363, 0.3511450381679389, 0.2396694214876033, 0.21296296296296297, 0.38650306748466257, 0.2767857142857143, 0.3592233009708738, 0.5299145299145299, 0.34, 0.4355044699872286, 0.3468208092485549, 0.23798882681564246, 0.3202614379084967, 0.3665594855305466, 0.4166666666666667, 0.2907801418439716, 0.26140808344198174, 0.3088235294117647, 0.3333333333333333, 0.35454545454545455, 0.32653061224489793, 0.3383084577114428, 0.37, 0.24096385542168675, 0.4093567251461988]})
#     """
#     metrics = defaultdict(list)
#     for subtask in leaderboard_task.subtasks:
#         for metric_name in leaderboard_task.metrics:
#             metric = get_metric(model_name, leaderboard_task.name, leaderboard_task.num_fewshot, leaderboard_task.use_cot, subtask, metric_name)
#             metrics[metric_name].append(metric)
#     return metrics

def get_leaderboard_aggregated_metrics(model_name, leaderboard_task):
    aggregated_metrics = EvalTaskResultInfo.from_evaltask(RESULTS_SAVE_ROOT, model_name, leaderboard_task).list_aggregated_metrics(allow_none=True)
    return aggregated_metrics

def get_tasks_aggregated_metrics(model_name: str, leaderboard_tasks: list[LeaderBoardTask]):
    metricname_to_val = dict(ChainMap(*[get_leaderboard_aggregated_metrics(model_name, task) for task in leaderboard_tasks]))
    return metricname_to_val
    

# def get_subtask_write_out_info(model_name, task_name, subtask_name, num_fewshot):
#     hashed_modelname = ModelInfo.get_decoded_modelname(model_name)
#     task_save_root = os.path.join(results_save_root,hashed_modelname,task_name)
#     writeout_infofile = os.path.join(task_save_root,str(get_lastest_tasks_version(model_name, task_name)),f'{num_fewshot}shot', f'{subtask_name}_write_out_info.json')

#     return open(writeout_infofile, 'r').readlines()

class EvalTaskResultInfo:
    """model + task + version + num_fewshot """
    def __init__(self,results_save_root: str, model_id:str, task_name: str, task_ver: int, num_fewshot :int, subtask_names: list[str], metric_names: list[str], metric_aggregate_ops: list[str], dataset_name: str) -> None:
        # eval task info
        self.results_save_root = results_save_root
        
        self.model_id = model_id
        self.task_name = task_name
        self.task_ver = task_ver
        self.num_fewshot = num_fewshot
        self.subtask_names = subtask_names
        
        self.metric_names = metric_names
        self.metric_aggregate_ops = metric_aggregate_ops
        self.dataset_name = dataset_name
        
        # more eval task info
        self.results_filepath = self._get_results_filepath()
        self.write_out_info_filepaths = self.list_write_out_info_filepaths(existing_files_only=True)
        
        self._metricname_to_aggregateop = dict(zip(self.metric_names, self.metric_aggregate_ops))
        
        # result info
        self._results = None
        pass
    
    @classmethod
    def from_evaltask(cls, results_save_root: str, model_id:str, leaderboardtask: LeaderBoardTask):
        obj = cls.__new__(cls)
        
        obj.results_save_root = results_save_root
        obj.model_id = model_id
        
        obj.task_name = leaderboardtask.name
        obj.task_ver = leaderboardtask.version
        obj.num_fewshot = leaderboardtask.num_fewshot
        obj.subtask_names = leaderboardtask.subtasks
        
        obj.metric_names = leaderboardtask.metrics
        obj.metric_aggregate_ops = leaderboardtask.aggregate_ops
        obj.dataset_name = leaderboardtask.dataset_name
        
        # more eval task info
        obj.results_filepath = obj._get_results_filepath()
        obj.write_out_info_filepaths = obj.list_write_out_info_filepaths(existing_files_only=True)
        
        obj._metricname_to_aggregateop = dict(zip(obj.metric_names, obj.metric_aggregate_ops))
        
        # result info
        obj._results = None
        
        return obj
        
    def __str__(self) -> str:
        """for print() str() format()"""
        return str({
            "model_id": self.model_id,
            "task_name": self.task_name,
            "task_ver": self.task_ver,
            "num_fewshot": self.num_fewshot,
            "subtask_names": self.subtask_names,
        })
    
    def is_result_exists(self):
        return Path(self.results_filepath).exists()
        
    def list_subtasknames(self):
        return self.subtask_names
    
    def _get_subtask_write_out_info_filepath(self, subtask_name):
        subtask_write_out_info_filepath = Path(self.results_save_root)/ModelInfo.encode_modelname(self.model_id)/self.task_name /str(self.task_ver)/f'{self.num_fewshot}shot'/f'{subtask_name}_write_out_info.json'
        return str(subtask_write_out_info_filepath)
    
    def list_write_out_info_filepaths(self, existing_files_only):
        write_out_info_filepaths = []
        for subtask_name in self.subtask_names:
            subtask_write_out_info_filepath=Path(self._get_subtask_write_out_info_filepath(subtask_name))
            if existing_files_only and not subtask_write_out_info_filepath.exists():
                continue
            else:
                write_out_info_filepaths.append(str(subtask_write_out_info_filepath))
        return write_out_info_filepaths
    
    def get_subtask_num_doc(self, subtask_name:str):
        """get task subject(vs generate) doc number to calculate aggregate metric"""
        
        cached_num_doc = self.search_num_of_docs_in_cache(subtask_name)
        if cached_num_doc:
            return cached_num_doc
    
        write_out_info_jsonpath = self._get_subtask_write_out_info_filepath( subtask_name)
        
        if not Path(write_out_info_jsonpath).exists():
            raise Exception(f"{write_out_info_jsonpath} not exists.")
        
        with open(write_out_info_jsonpath, 'r') as f:
            # dict of subtask name to dict of metricname to value
            write_out_info = json.load(f)
            num_doc = len(write_out_info)
            
            self.update_num_of_docs_cache(subtask_name, num_doc)
        return num_doc
    
    def get_subtask_pred_logits(self, subtask_name: str):
        write_out_info_jsonpath = self._get_subtask_write_out_info_filepath( subtask_name)
        
        records_logits = []
        with open(write_out_info_jsonpath, 'r') as f:
            # dict of subtask name to dict of metricname to value
            write_out_info = json.load(f)
            for item in write_out_info:
                item_logits = []
                for k, v in item.items():
                    if k.startswith("logit_"):
                        item_logits.append(v)
                        records_logits.append(item_logits)
        return records_logits
        
    def list_subtask_num_doc(self):

        subtasks = self.list_subtasknames()
        num_doc = {}
        for subtask in subtasks:
            num_doc[subtask] = self.get_subtask_num_doc(subtask)
        
        return 
    
    def list_subtask_logits(self):

        subtasks = self.list_subtasknames()
        logits = {}
        for subtask in subtasks:
            logits[subtask] = self.get_subtask_pred_logits(subtask)
        
        return logits
    
    def search_num_of_docs_in_cache(self, subtask_name):
        if Path(num_of_doc_cache_filepath).exists():
            try:
                with open(num_of_doc_cache_filepath, 'rb') as file:
                    num_of_doc_dict = pickle.load(file)
                    if num_of_doc_dict is None:
                        return None
                    if subtask_name in num_of_doc_dict:
                        return num_of_doc_dict[subtask_name]
                    else:
                        return None
            except FileNotFoundError:
                print("no recovery file.")
        else:
            return None
            
    def update_num_of_docs_cache(self, subtask_name, num_doc):
        """
        subtask:
            num_of_doc
        """
        try:
            # update
            if Path(num_of_doc_cache_filepath).exists():
                with open(num_of_doc_cache_filepath, 'rb') as file:
                    cached_num_of_doc_dict = pickle.load(file)
                    if cached_num_of_doc_dict is None:
                        cached_num_of_doc_dict = {}
                    if cached_num_of_doc_dict is None or subtask_name not in cached_num_of_doc_dict:
                        cached_num_of_doc_dict[subtask_name] = num_doc
            # create
            else:
                cached_num_of_doc_dict = { subtask_name: num_doc }
                
            with open(num_of_doc_cache_filepath, 'wb') as file:
                pickle.dump(cached_num_of_doc_dict, file)
        except FileNotFoundError:
            print("warning: not cache file specified.")
            
    def list_metricnames(self):
        """list the name of metrics"""
        return self.metric_names
    
    def _get_results_filepath(self):
        all_results_filepath = Path(self.results_save_root)/ModelInfo.encode_modelname(self.model_id)/self.task_name /str(self.task_ver)/f'{self.num_fewshot}shot'/'results.json'
        return str(all_results_filepath)
    
    def get_results(self, allow_none):
        """
        parse result per task
        returns: a dict contains subtasks result
        """
        if self._results is None:
            try:
                with open(self.results_filepath, 'r') as f:
                    # dict of subtask name to dict of metricname to value
                    self._results = json.load(f)['results']
            except FileNotFoundError:
                if allow_none:
                    return None
                else:
                    raise FileNotFoundError

        return self._results
        
    def get_subtask_metric(self, subtask_name, metric_name, allow_none:True):
        results = self.get_results(allow_none)
        if results is None: return None
        try:
          return results[subtask_name][metric_name]
        except KeyError or FileNotFoundError:
            if allow_none:
                return None
            else:
                raise KeyError   
            
    def _get_grouped_metricvals_by_subtask(self, allow_none:True):
        metrics = self.list_metrics(allow_none)
        grouped_metrics = defaultdict(dict)
        for subtask_name, metricname2val in metrics.items():
            for metric_name, vals in metricname2val.items():
                grouped_metrics[metric_name][subtask_name] = vals
        return grouped_metrics
    
    def get_aggregate_op(self, metric_name):
        return self._metricname_to_aggregateop[metric_name]
    
    def _aggregate_metric(self, aggregate_op: str, metric_vals: list, nums_of_doc):
        if aggregate_op == 'mean':
            metric_vals_sum = 0
            num_of_doc_sum = 0
            for (metric_val, num_of_doc) in zip(metric_vals, nums_of_doc):
                metric_vals_sum += metric_val * num_of_doc
                num_of_doc_sum += num_of_doc
            return metric_vals_sum / num_of_doc_sum
        else:
            raise Exception(f"aggregate op: {aggregate_op} is not supported.")
        
    def list_aggregated_metrics(self, allow_none: True):
        """aggregate metrics belongs to different subtasks, returns metric_name to aggregated val"""
        """each subtask should have metric while some metrics could be none"""
        grouped_metrics = self._get_grouped_metricvals_by_subtask(allow_none)
        aggregated_metricname2vals= {}
        for metric_name, subtaskname_to_val in grouped_metrics.items():
            aggragate_op = self.get_aggregate_op(metric_name)
            nums_of_doc = []
            vals = []
            
            # if all metrics is None, returns None
            if all([x == None for x in list(subtaskname_to_val.values())]):
                aggregated_metricname2vals[metric_name] = None
            else:
                # make sure all subtasks have got the eval metrics, otherwise print warining and returns None now. Another options is raise error.
                if not all([x != None for x in list(subtaskname_to_val.values())]):
                    print("[WARNING] partial of substask not get results")
                    aggregated_metricname2vals[metric_name] = None
                    break
                    # raise Exception("partial of substask not get results")
                elif len(subtaskname_to_val) == 1:
                    aggregated_metricname2vals[metric_name] = list(subtaskname_to_val.values())[0]
                else:
                    # do aggregate when subtasks count > 1
                    for subtask_name, val in subtaskname_to_val.items():
                        nums_of_doc.append(self.get_subtask_num_doc(subtask_name))
                        vals.append(val)
                        
                    aggregated_metricval = self._aggregate_metric(aggragate_op, vals, nums_of_doc)
                    aggregated_metricname2vals[metric_name] = aggregated_metricval
        return aggregated_metricname2vals
            
        
    def list_metrics(self, allow_none:True):
        """subtask metricname metricvalue"""
        metrics = defaultdict(dict)
        subtask_names = self.list_subtasknames()
        metric_names = self.list_metricnames()
        for subtask_name in subtask_names:
            for metric_name in metric_names:
                metric_val = self.get_subtask_metric(subtask_name, metric_name, allow_none)
                metrics[subtask_name][metric_name] = metric_val
        return dict(metrics)

if __name__ == "__main__":
    # print(list_models())
    # print(is_result_exists())
    
    sample_taskresultinfo = EvalTaskResultInfo.from_evaltask(RESULTS_SAVE_ROOT, 'huggyllama/llama-7b', LEADERBOARDTASK_REGISTRY['mmlu_likelihoodoptionkeycircular'])

    print(sample_taskresultinfo)

    # print(sample_taskresultinfo.list_metrics(allow_none=True))

    print(sample_taskresultinfo.list_aggregated_metrics(allow_none=True))



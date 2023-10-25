from autoeval_server.resultparser.loadresults import EvalTaskResultInfo, list_models
from config import RESULTS_SAVE_ROOT as results_save_root
from eval_and_dumping_result import LEADERBOARDTASK_REGISTRY
import numpy as np
import matplotlib.pyplot as plt
from itertools import chain
import pandas as pd

class PredLogitsStats:
    def __init__(self, result_info: EvalTaskResultInfo) -> None:
        self.result_info = result_info
        self.subtasks_logging_logits_dict = result_info.list_subtask_logits()
        self.logging_logits = chain.from_iterable(self.subtasks_logging_logits_dict.values())
        self.grouped_logging_logits = [self.group_options(np.array(x)) for x in self.logging_logits]
        self.aggregated_metrics = result_info.list_aggregated_metrics(allow_none=True)
        
    def group_options(self, options: np.array):
        num_options = int(np.sqrt(len(options)))
        return options.reshape((num_options, num_options)) 
        
    def stats_allsameoption_percentage(self, demical_places = 1):
        def all_same(items):
            return all(x == items[0] for x in items)
        
        isallsame = []
        for x in self.grouped_logging_logits:
            grouped_options_logits_argmax = np.array(x).argmax(-1)
            isallsame.append(all_same(grouped_options_logits_argmax))
        isallsame = np.array(isallsame)

        allsameoption_percentage = np.round(np.sum(isallsame.astype(int))/len(isallsame), demical_places+2)*100
        return allsameoption_percentage
    
    def stats_logits_std(self) -> np.array:
        return np.array(self.grouped_logging_logits).std(-1)
        
    def stats_choices_acc_circular_drop(self):
        if self.aggregated_metrics['next_token_argmax_choices_acc'] == 0:
            return 0
        else:
            return (self.aggregated_metrics['next_token_argmax_choices_acc'] - self.aggregated_metrics['next_token_argmax_choices_circular_acc'])/self.aggregated_metrics['next_token_argmax_choices_acc'] *100
        
    def stats_all_acc_circular_drop(self):
        if self.aggregated_metrics['next_token_argmax_all_acc'] == 0:
            return 0
        else:
            return (self.aggregated_metrics['next_token_argmax_all_acc'] - self.aggregated_metrics['next_token_argmax_all_circular_acc'])/self.aggregated_metrics['next_token_argmax_all_acc'] *100
    
    def stats_choices_acc_circular(self):
        return self.aggregated_metrics['next_token_argmax_choices_circular_acc']
    
    def stats_choices_acc(self):
        return self.aggregated_metrics['next_token_argmax_choices_acc']
    
    def stats_all(self):
        stats_attrs = [
            "allsameoption_percentage", 
            "options_std_avg", 
            "choices_acc_circular_drop", 
            "choices_acc_circular",
            "choices_acc"
            ]
        stats = {}
        for attr in stats_attrs:
            if attr == "allsameoption_percentage":
                stats[attr] = self.stats_allsameoption_percentage()
            if attr == "options_std_avg":
                stats[attr] = np.average(self.stats_logits_std())
            if attr == "choices_acc_circular_drop":
                stats[attr] = self.stats_choices_acc_circular_drop()
            if attr == "all_acc_circular_drop":
                stats[attr] = self.stats_all_acc_circular_drop()
            if attr == "choices_acc_circular":
                stats[attr] = self.stats_choices_acc_circular()
            if attr == "choices_acc":
                stats[attr] = self.stats_choices_acc()
        return stats

        
if __name__ == "__main__":
    
    models = list_models()
    tasks = [
        'hellaswag_likelihoodoptionkeycircular',
        'mmlu_likelihoodoptionkeycircular',
        # 'arc_challenge_likelihoodoptionkeycircular_0s',
        # 'arc_easy_likelihoodoptionkeycircular_0s'
        'boolq_likelihoodoptionkeycircular_0s',
        'piqa_likelihoodoptionkeycircular_0s',
        'siqa_likelihoodoptionkeycircular_0s'
    ]
    stats_dict = {}
    task = tasks[0]
    for model in models:
        try:
            sample_taskresultinfo = EvalTaskResultInfo.from_evaltask(results_save_root, model, LEADERBOARDTASK_REGISTRY[f'{task}'])
            statistician = PredLogitsStats(sample_taskresultinfo)
            stats = statistician.stats_all()
            stats_dict[model] = stats
        except FileNotFoundError:
            print(f"no result file for {model}, skip")
    
    print(stats_dict)
    pd_stats = pd.DataFrame.from_dict(stats_dict)
    print(pd_stats)
    pd_stats.to_pickle('my_alltasks_stats.pkl')
    
    pd_stats = pd.read_pickle('my_alltasks_stats.pkl')
    
    # from ydata_profiling import ProfileReport
    # profile = ProfileReport(pd_stats.T)

    # # Save report to a file
    # profile.to_file("profile_report.html")

    dict_stats = pd_stats.to_dict()
    print(pd_stats)
    # models = list(pd_stats.keys())
    # create scatter plot
    fig, ax1 = plt.subplots()
    scatter1 = ax1.scatter(list(pd_stats.loc['choices_acc_circular_drop']), list(pd_stats.loc['allsameoption_percentage']), color='red', label="allsame_percentage")
    ax1.tick_params('y', colors='red')
    
    ax2 = ax1.twinx()
    scatter2 = ax2.scatter(list(pd_stats.loc['choices_acc_circular_drop']), list(pd_stats.loc['options_std_avg']), color='blue', label="options_std_avg")
    ax1.tick_params('y', colors='blue')

    # Optionally set titles and labels
    # plt.title('Scatter plot of acc drop vs stats attr')
    # plt.xlabel('drop')
    # plt.ylabel('stats')
    
    
    # Handling the legend
    lines = [scatter1, scatter2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='best')
    # plt.legend(loc='best')

    # Display the plot
    # plt.show()
    plt.savefig("output_plot.png")
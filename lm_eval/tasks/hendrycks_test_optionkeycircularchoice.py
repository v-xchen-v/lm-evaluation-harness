"""
Measuring Massive Multitask Language Understanding
https://arxiv.org/pdf/2009.03300.pdf

The Hendryck's Test is a benchmark that measured a text model’s multitask accuracy.
The test covers 57 tasks including elementary mathematics, US history, computer
science, law, and more. To attain high accuracy on this test, models must possess
extensive world knowledge and problem solving ability. By comprehensively evaluating
the breadth and depth of a model’s academic and professional understanding,
Hendryck's Test can be used to analyze models across many tasks and to identify
important shortcomings.

Homepage: https://github.com/hendrycks/test
"""
from lm_eval.base import OptionKeyMultipleCircularChoiceTask
from lm_eval.tasks import hendrycks_test

_CITATION = hendrycks_test._CITATION

SUBJECTS = hendrycks_test.SUBJECTS

def create_all_tasks():
    """Creates a dictionary of tasks from a list of subjects
    :return: {task_name: task}
        e.g. {hendrycksTest-abstract_algebra_optionkeycircularchoice: Task, hendrycksTest-anatomy: Task}
    """
    return {f"hendrycksTest-{sub}_optionkeycircularchoice": create_task(sub) for sub in SUBJECTS}


def create_task(subject):
    class OptionKeyCircularChoiceHendrycksTest(OptionKeyCircularChoiceGeneralHendrycksTest):
        def __init__(self):
            super().__init__(subject)

    return OptionKeyCircularChoiceHendrycksTest


class OptionKeyCircularChoiceGeneralHendrycksTest(OptionKeyMultipleCircularChoiceTask):
    VERSION = 0
    DATASET_PATH = "cais/mmlu"
    DATASET_NAME = None

    def __init__(self, subject):
        self.DATASET_NAME = subject
        super().__init__()

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def validation_docs(self):
        return map(self._process_doc, self.dataset["validation"])

    def test_docs(self):
        return map(self._process_doc, self.dataset["test"])

    def format_example(self, doc, keys, circular_index=0):
        """
        circular 0:
        <prompt>
        A. <choice1>
        B. <choice2>
        C. <choice3>
        D. <choice4>
        Answer:

        circular 1:
        <prompt>
        A. <choice4>
        B. <choice1>
        C. <choice2>
        D. <choice3>
        Answer:
        """

        question = doc["question"].strip()
        if circular_index == 0:
            choices = "".join(
                [f"{key}. {choice}\n" for key, choice in zip(keys, doc["choices"])]
            )
        else:
            choices = ""
            for key_index, key in enumerate(keys):
                choices += f'{key}. {doc["choices"][(key_index-circular_index)%len(keys)]}\n'

        prompt = f"{question}\n{choices}Answer:"
        return prompt
    
    def _process_doc(self, doc):
        keys = ["A", "B", "C", "D"]
        return {
            "doc": doc,
            # "query": self.format_example(doc, keys),
            "choices": keys,
            "gold": doc["answer"],
        }

    def fewshot_examples(self, k, rnd):
        # fewshot_examples is not just sampling from train_docs because dev is
        # in the same distribution as val/test but auxiliary_train isn't
        if self._fewshot_docs is None:
            self._fewshot_docs = list(map(self._process_doc, self.dataset["dev"]))

        # use the unchanged order of the dev set without sampling,
        # just as in the original code https://github.com/hendrycks/test/blob/master/evaluate.py#L28
        return self._fewshot_docs[:k]

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["query"]


    
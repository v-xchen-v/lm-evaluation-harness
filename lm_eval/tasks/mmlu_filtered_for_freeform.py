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

from lm_eval.tasks import hendrycks_test
from lm_eval.tasks.hendrycks_test_likelihoodoptionkeycircular import LikelihoodOptionKeyMultipleCircularChoiceGeneralHendrycksTest
from lm_eval.tasks.hendrycks_test_likelihoodoptioncotent import LikelihoodOptionContentMultipleChoiceGeneralHendrycksTest

_CITATION = hendrycks_test._CITATION

class LikelihoodOptionKeyCircularMMLUFilteredForFreeformTest(LikelihoodOptionKeyMultipleCircularChoiceGeneralHendrycksTest):
    VERSION = 0
    DATASET_PATH = "v-xchen-v/mmlu_filtered_for_freeform"
    DATASET_NAME = None
    
    def __init__(self, subject=None):
        super().__init__(subject)
    
    def fewshot_context(self, doc, num_fewshot, **kwargs):
        subject = self.DATASET_NAME
        description = f"The following are multiple choice questions (with answers)."
        kwargs["description"] = description
        return super().fewshot_context(doc=doc, num_fewshot=num_fewshot, **kwargs)
    
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

        question = doc["prompt"].strip()
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
            "gold": keys.index(doc["answer"]),
        }
    
class LikelihoodOptionContentMMLUFilteredForFreeformTest(LikelihoodOptionContentMultipleChoiceGeneralHendrycksTest):
    VERSION = 0
    DATASET_PATH = "v-xchen-v/mmlu_filtered_for_freeform"
    DATASET_NAME = None
    
    def __init__(self, subject=None):
        super().__init__(subject)
    
    def fewshot_context(self, doc, num_fewshot, **kwargs):
        subject = self.DATASET_NAME
        description = f"Answer the following question in one line:"
        kwargs["description"] = description
        return super().fewshot_context(doc=doc, num_fewshot=num_fewshot, **kwargs)

    def _process_doc(self, doc):
        def format_example(doc, keys):
            """
            <prompt>
            question
            choice
            """

            question = doc["prompt"].strip()
            prompt = f"{question}\n"
            return prompt

        keys = ["A", "B", "C", "D"]
        return {
            "query": format_example(doc, keys),
            "choices": doc["choices"],
            "gold":  keys.index(doc["answer"]),
        }
    
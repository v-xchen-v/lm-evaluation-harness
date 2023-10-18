"""
WinoGrande: An Adversarial Winograd Schema Challenge at Scale
https://arxiv.org/pdf/1907.10641.pdf

WinoGrande is a collection of 44k problems, inspired by Winograd Schema Challenge
(Levesque, Davis, and Morgenstern 2011), but adjusted to improve the scale and
robustness against the dataset-specific bias. Formulated as a fill-in-a-blank
task with binary options, the goal is to choose the right option for a given
sentence which requires commonsense reasoning.

NOTE: This evaluation of Winogrande uses partial evaluation as described by
Trinh & Le in Simple Method for Commonsense Reasoning (2018).
See: https://arxiv.org/abs/1806.02847

Homepage: https://leaderboard.allenai.org/winogrande/submissions/public
"""

from lm_eval.base import LikelihoodOptionContentMultipleChoiceTask, LikelihoodOptionKeyMultipleCircularChoiceTask

_CITATION = """
@article{sakaguchi2019winogrande,
    title={WinoGrande: An Adversarial Winograd Schema Challenge at Scale},
    author={Sakaguchi, Keisuke and Bras, Ronan Le and Bhagavatula, Chandra and Choi, Yejin},
    journal={arXiv preprint arXiv:1907.10641},
    year={2019}
}
"""

class WinograndeCustom:
    DATASET_PATH = "winogrande"
    DATASET_NAME = "winogrande_xl"
    
    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(map(self._process_doc, self.dataset["train"]))
        return self._training_docs

    def validation_docs(self):
        return map(self._process_doc, self.dataset["validation"])
    
    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["sentence"]
    
    @classmethod
    def partial_context(cls, doc, option):
        # Substitute the pronoun in the sentence with the specified option
        # and ignore everything after.
        pronoun_loc = doc["sentence"].index("_")
        return doc["sentence"][:pronoun_loc] + option
    
    @classmethod
    def partial_target(cls, doc):
        # The target is everything after the document specified pronoun.
        pronoun_loc = doc["sentence"].index("_") + 1
        return " " + doc["sentence"][pronoun_loc:].strip()
    
class WinograndeCustomOptionContent(WinograndeCustom, LikelihoodOptionContentMultipleChoiceTask):
    VERSION = 0
    def _process_doc(self, doc):
        out_doc = {
            "query": f"Question: {doc['sentence'].rstrip('.')}?\nAnswer:",
            "choices": [doc["option1"], doc["option2"]],
            "gold": int(doc['answer'])-1 # answer_to_num = {"1": 0, "2": 1}
        }
        return out_doc
    
    def doc_to_text(self, doc):
        return doc["query"]
    
class WinograndeCustomOptionKeyCircular(WinograndeCustom, LikelihoodOptionKeyMultipleCircularChoiceTask):
    VERSION = 0
    
    def format_example(self, doc, keys, circular_index=0):
        """
        circular 0:
        <prompt>
        A. <choice1>
        B. <choice2>
        Answer:

        circular 1:
        <prompt>
        A. <choice2>
        B. <choice1>
        Answer:
        """

        choice_texts = [doc["option1"], doc["option2"]]
        if circular_index == 0:
            choices = "".join(
                [f"{key}. {choice}\n" for key, choice in zip(keys, choice_texts)]
            )
        else:
            choices = ""
            for key_index, key in enumerate(keys):
                choices += f'{key}. {choice_texts[(key_index-circular_index)%len(keys)]}\n'

        # query prompt
        prompt = f"Question: {doc['sentence'].rstrip('.')}?\n{choices}Answer:"
        return prompt
    
        
    def _process_doc(self, doc):
        out_doc = {
            "sentence": doc["sentence"],
            "choices": ["A", "B"],
            "gold": int(doc['answer'])-1, # answer_to_num = {"1": 0, "2": 1}
            "doc": doc,
        }
        return out_doc
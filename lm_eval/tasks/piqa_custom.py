"""
PIQA: Reasoning about Physical Commonsense in Natural Language
https://arxiv.org/pdf/1911.11641.pdf

Physical Interaction: Question Answering (PIQA) is a physical commonsense
reasoning and a corresponding benchmark dataset. PIQA was designed to investigate
the physical knowledge of existing models. To what extent are current approaches
actually learning about the world?

Homepage: https://yonatanbisk.com/piqa/
"""
from lm_eval.base import LikelihoodOptionContentMultipleChoiceTask, LikelihoodOptionKeyMultipleCircularChoiceTask


_CITATION = """
@inproceedings{Bisk2020,
    author = {Yonatan Bisk and Rowan Zellers and
            Ronan Le Bras and Jianfeng Gao
            and Yejin Choi},
    title = {PIQA: Reasoning about Physical Commonsense in
           Natural Language},
    booktitle = {Thirty-Fourth AAAI Conference on
               Artificial Intelligence},
    year = {2020},
}
"""

class PIQACustom:
    DATASET_PATH = "piqa"
    DATASET_NAME = None

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
    
class PIQACustomOptionContent(PIQACustom, LikelihoodOptionContentMultipleChoiceTask):
    VERSION = 0
    def _process_doc(self, doc):
        out_doc = {
            "query": "Question: " + doc["goal"] + "\nAnswer:",
            "choices": [doc["sol1"], doc["sol2"]],
            "gold": int(doc['label'])
        }
        return out_doc
    
    def doc_to_text(self, doc):
        return doc["query"]
    
class PIQACustomOptionKeyCircular(PIQACustom, LikelihoodOptionKeyMultipleCircularChoiceTask):
    VERSION = 0
    SUBJECT = "piqa"
    
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

        choice_texts = [doc["sol1"], doc["sol2"]]
        if circular_index == 0:
            choices = "".join(
                [f"{key}. {choice}\n" for key, choice in zip(keys, choice_texts)]
            )
        else:
            choices = ""
            for key_index, key in enumerate(keys):
                choices += f'{key}. {choice_texts[(key_index-circular_index)%len(keys)]}\n'

        # query prompt
        prompt = f"Question: {doc['goal']}\n{choices}Answer:"
        return prompt
    
    def _process_doc(self, doc):
        keys = ["A", "B"]
        out_doc = {
            "choices": keys,
            "gold": int(doc['label']),
            "doc": doc,
        }
        return out_doc  
    
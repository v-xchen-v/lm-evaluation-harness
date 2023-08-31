"""
HellaSwag: Can a Machine Really Finish Your Sentence?
https://arxiv.org/pdf/1905.07830.pdf

Hellaswag is a commonsense inference challenge dataset. Though its questions are
trivial for humans (>95% accuracy), state-of-the-art models struggle (<48%). This is
achieved via Adversarial Filtering (AF), a data collection paradigm wherein a
series of discriminators iteratively select an adversarial set of machine-generated
wrong answers. AF proves to be surprisingly robust. The key insight is to scale up
the length and complexity of the dataset examples towards a critical 'Goldilocks'
zone wherein generated text is ridiculous to humans, yet often misclassified by
state-of-the-art models.

Homepage: https://rowanzellers.com/hellaswag/
"""
import re
from lm_eval.base import MultipleCircularChoiceTask
import numpy as np
from lm_eval.metrics import mean

_CITATION = """
@inproceedings{zellers2019hellaswag,
    title={HellaSwag: Can a Machine Really Finish Your Sentence?},
    author={Zellers, Rowan and Holtzman, Ari and Bisk, Yonatan and Farhadi, Ali and Choi, Yejin},
    booktitle ={Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics},
    year={2019}
}
"""

class HellaSwagCircularChoice(MultipleCircularChoiceTask):
    VERSION = 0
    DATASET_PATH = "hellaswag"
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
        
        question = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
        choices = [self.preprocess(ending) for ending in doc["endings"]]
        if circular_index == 0:
            options = "".join(
                [f"{key}. {choice}\n" for key, choice in zip(keys, choices)]
            )
        else:
            options = ""
            for key_index, key in enumerate(keys):
                options += f'{key}. {choices[(key_index-circular_index)%len(keys)]}\n'
        ctx = f"{question}\n{options}Answer:"
        prompt = self.preprocess(doc["activity_label"] + ": " + ctx)
        return prompt
    
    def _process_doc(self, doc):
        # ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
        keys = ["A", "B", "C", "D"]
        out_doc = {
            # "query": self.preprocess(doc["activity_label"] + ": " + ctx),
            "doc": doc,
            "choices": keys,
            "gold": int(doc["label"]),
        }
        return out_doc

    @classmethod
    def preprocess(cls, text):
        text = text.strip()
        # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
        text = text.replace(" [title]", ". ")
        text = re.sub("\\[.*?\\]", "", text)
        text = text.replace("  ", " ")
        return text

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["query"]
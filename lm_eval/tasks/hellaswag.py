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
from lm_eval.base import MultipleChoiceTask
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


class HellaSwag(MultipleChoiceTask):
    VERSION = 1
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

    def _process_doc(self, doc):
        ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
        out_doc = {
            "query": self.preprocess(doc["activity_label"] + ": " + ctx),
            "choices": [self.preprocess(ending) for ending in doc["endings"]],
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

    def doc_to_text(self, doc):
        return doc["query"]

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["query"]

    def process_results(self, doc, results):
        gold = doc["gold"]

        logits = [x[0] for x in results]
        max_equals = [x[1] for x in results]

        # the one with biggest probility is match the label, treat is as right
        acc_choicesonly = 1.0 if np.argmax(logits) == gold else 0.0

        # the one fully match the label, treat it as right. If it has higgest probility but not exactly match treat it as not right.
        em = 1.0 if max_equals[gold] else 0.0
        completion_len = np.array([float(len(i)) for i in doc["choices"]])
        acc_norm = 1.0 if np.argmax(logits / completion_len) == gold else 0.0

        return {
            "acc": acc_choicesonly,
            "acc_norm": acc_norm,
            "em": em
        }

    def aggregation(self):
        return {
            "acc": mean,
            "acc_norm": mean,
            "em": mean
        }
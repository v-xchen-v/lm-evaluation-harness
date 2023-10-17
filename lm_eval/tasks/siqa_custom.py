"""
SocialIQA: Commonsense Reasoning about Social Interactions
https://arxiv.org/pdf/1904.09728.pdf

We introduce Social IQa, the first largescale benchmark for commonsense reasoning about social situations. Social IQa contains 38,000 multiple choice questions for probing emotional and social intelligence in a variety of everyday situations (e.g., Q: "Jordan wanted to tell Tracy a secret, so Jordan leaned towards Tracy. Why did Jordan do this?" A: "Make sure no one else could hear"). Through crowdsourcing, we collect commonsense questions along with correct and incorrect answers about social interactions, using a new framework that mitigates stylistic artifacts in incorrect answers by asking workers to provide the right answer to a different but related question. Empirical results show that our benchmark is challenging for existing question-answering models based on pretrained language models, compared to human performance (>20% gap). Notably, we further establish Social IQa as a resource for transfer learning of commonsense knowledge, achieving state-of-the-art performance on multiple commonsense reasoning tasks (Winograd Schemas, COPA).

Homepage: https://leaderboard.allenai.org/socialiqa/submissions/public
"""

from lm_eval.base import LikelihoodOptionContentMultipleChoiceTask, LikelihoodOptionKeyMultipleCircularChoiceTask

_CITATION = """
@inproceedings{sap2019socialiqa,
    title="{SocialIQA}: Commonsense Reasoning about Social Interactions",
    author="Maarten Sap and Hannah Rashkin and Derek Chen and Ronan LeBras and Yejin Choi",
    year="2019",
    booktitle="EMNLP"
}
"""

class SIQACustom:
    DATASET_PATH = "lighteval/siqa"
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
    
    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["context"]
    
class SIQACustomOptionContent(SIQACustom, LikelihoodOptionContentMultipleChoiceTask):
    VERSION = 0
    
    def _process_doc(self, doc):
        out_doc = {
            "query": f"{doc['context']}\nQuestion: {doc['question']}\nAnswer:",
            "choices": [doc['answerA'], doc['answerB'], doc['answerC']],
            "gold": int(doc['label'])-1 # label: 1, 2, 3 is stands for first, second and third candidate answer.
        }
        return out_doc
    
    def doc_to_text(self, doc):
        return doc["query"]
    
class SIQACustomOptionKeyCircular(SIQACustom, LikelihoodOptionKeyMultipleCircularChoiceTask):
    VERSION = 0
    SUBJECT = 'siqa'
    
    def format_example(self, doc, keys, circular_index=0):
        """
        circular 0:
        <prompt>
        A. <choice1>
        B. <choice2>
        C. <choice3>
        Answer:

        circular 1:
        <prompt>
        A. <choice2>
        B. <choice3>
        C. <choice1>
        Answer:
        
        circular 2:
        <prompt>
        A. <choice3>
        B. <choice1>
        B. <choice2>
        Answer:
        """

        choice_texts = [doc['answerA'], doc['answerB'], doc['answerC']]
        if circular_index == 0:
            choices = "".join(
                [f"{key}. {choice}\n" for key, choice in zip(keys, choice_texts)]
            )
        else:
            choices = ""
            for key_index, key in enumerate(keys):
                choices += f'{key}. {choice_texts[(key_index-circular_index)%len(keys)]}\n'

        # query prompt
        prompt = f"{doc['context']}\nQuestion: {doc['question']}\n{choices}Answer:"
        return prompt
    
    def _process_doc(self, doc):
        out_doc = {
            "context": doc["context"],
            "choices": ["A", "B", "C"],
            "gold": int(doc['label'])-1,
            "doc": doc,
        }
        return out_doc   
    
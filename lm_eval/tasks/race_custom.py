"""
RACE: Large-scale ReAding Comprehension Dataset From Examinations
https://arxiv.org/pdf/1704.04683.pdf

RACE is a large-scale reading comprehension dataset with more than 28,000 passages
and nearly 100,000 questions. The dataset is collected from English examinations
in China, which are designed for middle school and high school students. The dataset
can be served as the training and test sets for machine comprehension.

Homepage: https://www.cs.cmu.edu/~glai1/data/race/
"""

from lm_eval.base import LikelihoodOptionContentMultipleChoiceTask, LikelihoodOptionKeyMultipleCircularChoiceTask

_CITATION = """
@article{lai2017large,
    title={RACE: Large-scale ReAding Comprehension Dataset From Examinations},
    author={Lai, Guokun and Xie, Qizhe and Liu, Hanxiao and Yang, Yiming and Hovy, Eduard},
    journal={arXiv preprint arXiv:1704.04683},
    year={2017}
}
"""

class RACECustom:
    DATASET_PATH = "race"
     
    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True
    
    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(map(self._process_doc,self.dataset["train"]))
        return self._training_docs

    def test_docs(self):
        return map(self._process_doc, self.dataset["test"])
    
    def validation_docs(self):
        return map(self._process_doc, self.dataset["validation"])
    
    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["article"]
    
class RACEHighCustomOptionContent(RACECustom, LikelihoodOptionContentMultipleChoiceTask):
    DATASET_NAME = "high"
    VERSION = 0
    
    # prompt: Article:\n{article}\n\nQuestion: {question}\nAnswer:
    def _process_doc(self, doc):
        out_doc = {
            "id": doc["example_id"],
            "query": f"Article:\n{doc['article']}\n\nQuestion: {doc['question']}\nAnswer:",
            "choices": doc["options"],
            "gold": ["A", "B", "C", "D"].index(doc['answer'].strip())
        }
        return out_doc
    
    def doc_to_text(self, doc):
        return doc["query"]
    
class RACEMiddleCustomOptionContent(RACEHighCustomOptionContent):
    DATASET_NAME = "middle"
    
class RACEAllCustomOptionContent(RACEHighCustomOptionContent):
    DATASET_NAME = "all"
    
class RACEHighCustomOptionKeyCircular(RACECustom, LikelihoodOptionKeyMultipleCircularChoiceTask):
    DATASET_NAME = "high"
    VERSION = 0
    
    def fewshot_context(self, doc, num_fewshot, **kwargs):
        description = f'The following are multiple choice questions (with answers).'
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
        B. <choice3>
        C. <choice2>
        D. <choice1>
        Answer:
        
        ...
        """

        choice_texts = doc["options"]
        if circular_index == 0:
            choices = "".join(
                [f"{key}. {choice}\n" for key, choice in zip(keys, choice_texts)]
            )
        else:
            choices = ""
            for key_index, key in enumerate(keys):
                choices += f'{key}. {choice_texts[(key_index-circular_index)%len(keys)]}\n'

        # query prompt: Article:\n{article}\n\nQuestion: {question}\n\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer:
        prompt = f"Article:\n{doc['article']}\n\nQuestion: {doc['question']}\n{choices}Answer:"
        return prompt
    
    def _process_doc(self, doc):
        keys = ["A", "B", "C", "D"]
        out_doc = {
            "id": doc["example_id"],
            "choices": keys,
            "gold": keys.index(doc["answer"].strip()),
            "doc": doc,
        }
        return out_doc
    
class RACEMiddleCustomOptionKeyCircular(RACEHighCustomOptionKeyCircular):
    DATASET_NAME = "middle"
    
class RACEAllCustomOptionKeyCircular(RACEHighCustomOptionKeyCircular):
    DATASET_NAME = "all"
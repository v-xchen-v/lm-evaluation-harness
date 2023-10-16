"""
SuperGLUE: A Stickier Benchmark for General-Purpose Language Understanding Systems
https://w4ngatang.github.io/static/papers/superglue.pdf

SuperGLUE is a benchmark styled after GLUE with a new set of more difficult language
understanding tasks.

Homepage: https://super.gluebenchmark.com/

"""
from lm_eval.base import rf, Task, LikelihoodOptionContentMultipleChoiceTask, LikelihoodOptionKeyMultipleCircularChoiceTask

_CITATION = """
@inproceedings{NEURIPS2019_4496bf24,
    author = {Wang, Alex and Pruksachatkun, Yada and Nangia, Nikita and Singh, Amanpreet and Michael, Julian and Hill, Felix and Levy, Omer and Bowman, Samuel},
    booktitle = {Advances in Neural Information Processing Systems},
    editor = {H. Wallach and H. Larochelle and A. Beygelzimer and F. d\textquotesingle Alch\'{e}-Buc and E. Fox and R. Garnett},
    pages = {},
    publisher = {Curran Associates, Inc.},
    title = {SuperGLUE: A Stickier Benchmark for General-Purpose Language Understanding Systems},
    url = {https://proceedings.neurips.cc/paper/2019/file/4496bf24afe7fab6f046bf4923da8de6-Paper.pdf},
    volume = {32},
    year = {2019}
}
"""

class BoolQCustom(Task):
    DATASET_PATH = "super_glue"
    DATASET_NAME = "boolq"
    
    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False
    
    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(map(self._process_doc,self.dataset["train"]))
        return self._training_docs

    def validation_docs(self):
        return map(self._process_doc, self.dataset["validation"])
    
    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["passage"]

class BoolQCustomOptionContent(BoolQCustom, LikelihoodOptionContentMultipleChoiceTask):
    VERSION = 0
    def _process_doc(self, doc):
        out_doc = {
            "id": doc["idx"],
            "query": f"{doc['passage']}\nQuestion: {doc['question']}?\nAnswer:",
            "choices": ["no", "yes"],
            "gold": int(doc['label'])
        }
        return out_doc
    
    def doc_to_text(self, doc):
        return doc["query"]
    
class BoolQCustomOptionKeyCircular(BoolQCustom, LikelihoodOptionKeyMultipleCircularChoiceTask):
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

        choice_texts = ["no", "yes"]
        if circular_index == 0:
            choices = "".join(
                [f"{key}. {choice}\n" for key, choice in zip(keys, choice_texts)]
            )
        else:
            choices = ""
            for key_index, key in enumerate(keys):
                choices += f'{key}. {choice_texts[(key_index-circular_index)%len(keys)]}\n'

        # query prompt
        prompt = f"{doc['passage']}\nQuestion: {doc['question']}?\n{choices}Answer:"
        return prompt
    
    def _process_doc(self, doc):
        out_doc = {
            "id": doc["idx"],
            "passage": doc["passage"],
            "choices": ["A", "B"],
            "gold": int(doc['label']),
            "doc": doc,
        }
        return out_doc
"""
AGIEval: A Human-Centric Benchmark for Evaluating Foundation Models
https://arxiv.org/pdf/2304.06364.pdf

AGIEval is a human-centric benchmark specifically designed to evaluate the general abilities of foundation models in tasks pertinent to human cognition and problem-solving. This benchmark is derived from 20 official, public, and high-standard admission and qualification exams intended for general human test-takers, such as general college admission tests (e.g., Chinese College Entrance Exam (Gaokao) and American SAT), law school admission tests, math competitions, lawyer qualification tests, and national civil service exams.

Homepage: https://github.com/microsoft/AGIEval
"""
from lm_eval.base import LikelihoodOptionKeyMultipleCircularChoiceTask, LikelihoodOptionContentMultipleChoiceTask

_CITATION = """
@misc{zhong2023agieval,
      title={AGIEval: A Human-Centric Benchmark for Evaluating Foundation Models}, 
      author={Wanjun Zhong and Ruixiang Cui and Yiduo Guo and Yaobo Liang and Shuai Lu and Yanlin Wang and Amin Saied and Weizhu Chen and Nan Duan},
      year={2023},
      eprint={2304.06364},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""

SUBJECTS = [
    "lsat-ar",
    "lsat-lr",
    "lsat-rc",
    "logiqa-en",
    "sat-math",
    "sat-en",
    "aqua-rat",
    "sat-en-without-passage",
    # "gaokao-english"
]

def create_all_optionkeycircular_tasks():
    """Creates a dictionary of task from a list of subjects
    :return: {task_name: task}
        e.g. {agieval_eng_qa_lsat-ar_optionkeycircular: Task, agieval_eng_qa_lsat-lr_optionkeycircular: Task}
    """
    return {f"agieval_eng_qa_{sub}_likelihoodoptionkeycircular": create_optionkeycircular_task(sub) for sub in SUBJECTS}

def create_all_optioncontent_tasks():
    """Creates a dictionary of task from a list of subjects
    :return: {task_name: task}
        e.g. {agieval_eng_qa_lsat-ar_optioncontent: Task, agieval_eng_qa_lsat-lr_optioncontent: Task}
    """
    return {f"agieval_eng_qa_{sub}_likelihoodoptioncontent": create_optioncontent_task(sub) for sub in SUBJECTS}

def create_optionkeycircular_task(subject):
    class AGIEvalCustomOptionKeyCircularEngQA(GeneralAGIEvalCustomEngQAOptionKeyCircular):
        def __init__(self):
            super().__init__(subject)

    return AGIEvalCustomOptionKeyCircularEngQA

def create_optioncontent_task(subject):
    class AGIEvalCustomOptionContentEngQA(GeneralAGIEvalCustomEngQAOptionContent):
        def __init__(self):
            super().__init__(subject)

    return AGIEvalCustomOptionContentEngQA

class GeneralAGIEvalCustomEngQA:
    DATASET_PATH = "v-xchen-v/agieval_eng_qa"
    DATASET_NAME = None
    
    def __init__(self, subject):
        self.DATASET_NAME = subject
        super().__init__()

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False
    
    def validation_docs(self):
        if self.has_validation_docs():
            return map(self._process_doc, self.dataset["validation"])

 
    
class GeneralAGIEvalCustomEngQAOptionKeyCircular(GeneralAGIEvalCustomEngQA, LikelihoodOptionKeyMultipleCircularChoiceTask):
    VERSION = 0

    def format_example(self, doc, keys, circular_index=0):
        """
        formating query prompt
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
        
        choice_texts = [option[3:] for option in doc['options']] # remove (A), (B), ... at the beginning
        keys = keys[: len(choice_texts)] # handle the data issue that some row has fewer options
        if circular_index == 0:
            choices = "".join(
                [f"{choice_key}. {choice_text}\n" for choice_key, choice_text in zip(keys, choice_texts)]
            )
        else:
            choices = ""
            for key_index, key in enumerate(keys):
                choices += f'{key}. {choice_texts[(key_index-circular_index)%len(keys)]}\n'
        
        query_prompt = ("" if doc["passage"] is None else doc["passage"]) + "Q: " +   doc["question"] + " " + "Answer Choices: " + choices + f"A: Among A through {keys[-1]}, the answer is"
        
        # The query prompt portion of the document example.
        return query_prompt
        
    def _process_doc(self, doc):
        if self.DATASET_NAME in ("lsat-ar", "lsat-lr", "lsat-rc", "aqua-rat"):
            choice_keys = ["A", "B", "C", "D", "E"]
        else:
            choice_keys = ["A", "B", "C", "D"]
            
        return {
            "doc": doc, # The doc to generate query prompt.
            "choices": choice_keys,  # The list of choices.
            "gold": int(doc["label"]),  # The integer used to index into the correct element of `"choices"`.
        }
        
    def fewshot_examples(self, k):
        if self._fewshot_docs is None:
            self._fewshot_docs = list(map(self._process_doc, self.dataset['dev']))

        return self._fewshot_docs[:min(k, len(self._fewshot_docs))]

    def fewshot_context(self, doc, num_fewshot, circular_index, provide_description=None, rnd=None, description=None):
        """Returns a fewshot context string that is made up of a prepended description
        (if provided), the `num_fewshot` number of examples, and an appended prompt example.

        :param doc: str
            The document as returned from training_docs, validation_docs, or test_docs.
        :param num_fewshot: int
            The number of fewshot examples to provide in the returned context string.
        :param provide_description: bool
            Not implemented, and this option is deprecated and will be removed in a future version in favor of a different description providing method
        :param rnd: random.Random
            The pseudo-random number generator used to randomly sample examples.
            WARNING: This is currently a required arg although it's optionalized with a default `None`.
        :param description: str
            The task's description that will be prepended to the fewshot examples.
        :returns: str
            The fewshot context.
        """
        assert (
            rnd is not None
        ), "A `random.Random` generator argument must be provided to `rnd`"
        assert not provide_description, (
            "The `provide_description` arg will be removed in future versions. To prepend "
            "a custom description to the context, supply the corresponding string via the "
            "`description` arg."
        )
        
        description = f"Here are the answers for the problems in the exam."
        description = description + "\n\n" if description else ""

        if num_fewshot == 0:
            labeled_examples = ""
        else:
            fewshotex = self.fewshot_examples(k=num_fewshot)

            labeled_examples = ""
            for fewshot_idx, doc in enumerate(fewshotex):
                  "Problem {}.   ".format(fewshot_idx + 1) + self.doc_to_text(doc) + self.doc_to_target(doc) + "\n\n"

        example = self.doc_to_text(doc, circular_index)
        return description + labeled_examples + example
    
class GeneralAGIEvalCustomEngQAOptionContent(GeneralAGIEvalCustomEngQA, LikelihoodOptionContentMultipleChoiceTask):
    VERSION = 0

    def _process_doc(self, doc):
        out_doc = {
            "query": ("" if doc["passage"] is None else doc["passage"]) + "\n" + "Q: " +   doc["question"] + "\n" + "A:",
            "choices": [option[3:] for option in doc['options']], # option[3:] to remove (A), (B), ... at the beginning
            "gold": int(doc['label'])
        }
        return out_doc
    
    def doc_to_text(self, doc):
        return doc["query"]
    
    def fewshot_context(self, doc, num_fewshot, provide_description=None, rnd=None, description=None):
        """Returns a fewshot context string that is made up of a prepended description
        (if provided), the `num_fewshot` number of examples, and an appended prompt example.

        :param doc: str
            The document as returned from training_docs, validation_docs, or test_docs.
        :param num_fewshot: int
            The number of fewshot examples to provide in the returned context string.
        :param provide_description: bool
            Not implemented, and this option is deprecated and will be removed in a future version in favor of a different description providing method
        :param rnd: random.Random
            The pseudo-random number generator used to randomly sample examples.
            WARNING: This is currently a required arg although it's optionalized with a default `None`.
        :param description: str
            The task's description that will be prepended to the fewshot examples.
        :returns: str
            The fewshot context.
        """
        assert (
            rnd is not None
        ), "A `random.Random` generator argument must be provided to `rnd`"
        assert not provide_description, (
            "The `provide_description` arg will be removed in future versions. To prepend "
            "a custom description to the context, supply the corresponding string via the "
            "`description` arg."
        )
        
        description = f"Here are the answers for the problems in the exam."
        description = description + "\n\n" if description else ""

        if num_fewshot == 0:
            labeled_examples = ""
        else:
            fewshotex = self.fewshot_examples(k=num_fewshot)

            labeled_examples = ""
            for fewshot_idx, doc in enumerate(fewshotex):
                  "Problem {}.   ".format(fewshot_idx + 1) + self.doc_to_text(doc) + self.doc_to_target(doc) + "\n\n"

        example = self.doc_to_text(doc)
        return description + labeled_examples + example
"""
AGIEval: A Human-Centric Benchmark for Evaluating Foundation Models
https://arxiv.org/pdf/2304.06364.pdf

AGIEval is a human-centric benchmark specifically designed to evaluate the general abilities of foundation models in tasks pertinent to human cognition and problem-solving. This benchmark is derived from 20 official, public, and high-standard admission and qualification exams intended for general human test-takers, such as general college admission tests (e.g., Chinese College Entrance Exam (Gaokao) and American SAT), law school admission tests, math competitions, lawyer qualification tests, and national civil service exams.

Homepage: https://github.com/microsoft/AGIEval
"""
from lm_eval.base import MultipleChoiceTask

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
    "gaokao-english"
]

def create_all_tasks():
    """Creates a dictionary of task from a list of subjects
    :return: {task_name: task}
        e.g. {agieval_eng_qa_lsat-ar: Task, agieval_eng_qa_lsat-lr: Task}
    """
    return {f"agieval_eng_qa_{sub}": create_task(sub) for sub in SUBJECTS}

def create_task(subject):
    class AGIEvalEngQA(GeneralAGIEvalEngQA):
        def __init__(self):
            super().__init__(subject)

    return AGIEvalEngQA

class GeneralAGIEvalEngQA(MultipleChoiceTask):
    VERSION = 1
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

    def training_docs(self):
        if self.has_training_docs():
            if self._training_docs is None:
                self._training_docs = list(
                    map(self._process_doc, self.dataset["train"])
                )
            return self._training_docs

    def validation_docs(self):
        if self.has_validation_docs():
            return map(self._process_doc, self.dataset["validation"])

    def test_docs(self):
        if self.has_test_docs():
            # Return the test document generator from `self.dataset`.
            # In most case you can leave this as is unless the dataset split is
            # named differently than the default `"test"`.
            return map(self._process_doc, self.dataset["test"])

    def _process_doc(self, doc):
        prompt = ("" if doc["passage"] is None else doc["passage"]) + "Q: " + doc["question"] + " " \
            + "Answer Choices: " + " ".join(doc["options"]) + "\n" + \
            "A: Among A through E, the answer is"
        choice_names = ["A", "B", "C", "D", "E"]

        return {
            "query": prompt,  # The query prompt.
            "choices": choice_names,  # The list of choices.
            "gold": doc["label"],  # The integer used to index into the correct element of `"choices"`.
        }

    def doc_to_text(self, doc):
        # The query prompt portion of the document example.
        return doc["query"]

    def fewshot_examples(self, k):
        if self._fewshot_docs is None:
            self._fewshot_docs = list(map(self._process_doc, self.dataset['dev']))

        return self._fewshot_docs[:min(k, len(self._fewshot_docs))]

    def fewshot_context(
        self, doc, num_fewshot, provide_description=None, rnd=None, description=None
    ):
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
        if provide_description is not None:
            # nudge people to not specify it at all
            print(
                "WARNING: provide_description is deprecated and will be removed in a future version in favor of description_dict"
            )

        description = description + "\n\n" if description else ""

        if num_fewshot == 0:
            labeled_examples = ""
        else:
            fewshotex = self.fewshot_examples(k=num_fewshot)

            labeled_examples = (
                "\n\n".join(
                    [
                        self.doc_to_text(doc) + self.doc_to_target(doc)
                        for doc in fewshotex
                    ]
                )
                + "\n\n"
            )

        example = self.doc_to_text(doc)
        return description + labeled_examples + example

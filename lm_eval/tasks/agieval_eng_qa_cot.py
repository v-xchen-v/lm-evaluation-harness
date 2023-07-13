"""
AGIEval: A Human-Centric Benchmark for Evaluating Foundation Models
https://arxiv.org/pdf/2304.06364.pdf

AGIEval is a human-centric benchmark specifically designed to evaluate the general abilities of foundation models in tasks pertinent to human cognition and problem-solving. This benchmark is derived from 20 official, public, and high-standard admission and qualification exams intended for general human test-takers, such as general college admission tests (e.g., Chinese College Entrance Exam (Gaokao) and American SAT), law school admission tests, math competitions, lawyer qualification tests, and national civil service exams.

Homepage: https://github.com/microsoft/AGIEval
"""
from lm_eval.tasks.agieval_eng_qa import GeneralAGIEvalEngQA
from lm_eval.tasks.agieval_eng_qa import _CITATION, SUBJECTS
import torch

_CITATION = _CITATION

SUBJECTS = SUBJECTS

def create_all_tasks():
    """Creates a dictionary of task from a list of subjects
    :return: {task_name: task}
        e.g. {agieval_eng_qa_cot_lsat-ar: Task, agieval_eng_qa_cot_lsat-lr: Task}
    """
    return {f"agieval_eng_qa_cot_{sub}": create_task(sub) for sub in SUBJECTS}

def create_task(subject):
    class AGIEvalEngQACoT(GeneralAGIEvalEngQACoT):
        def __init__(self):
            super().__init__(subject)

    return AGIEvalEngQACoT

class GeneralAGIEvalEngQACoT(GeneralAGIEvalEngQA):
    VERSION = 1
    
    def doc_to_zeroshot_prompt(self, doc, lm, max_length=None):
        stage1_prompt = ("" if doc["passage"] is None else doc["passage"]) + "Q: " +   doc["question"] + " " \
            + "Answer Choices: " + " ".join(doc["options"]) + "\n" + \
            "Let's think step by step."
        generated_explanation = self.call_model_generate(lm, stage1_prompt, None, max_length)
        prompt = self.generate_second_stage_input(stage1_prompt, generated_explanation)
        return prompt
    
    def call_model_generate(self, lm, context, until, max_length):   
        if isinstance(until, str):
            until = [until]
        stop_sequences = until
        max_generation_length = max_length

        assert (
                isinstance(max_generation_length, int) or max_generation_length is None
            )
        assert isinstance(stop_sequences, list) or stop_sequences is None

        # TODO: Find a better way to handle stop sequences for 0-shot.
        if stop_sequences is None:
            until = [lm.eot_token]
        else:
            until = stop_sequences + [lm.eot_token]

        if max_generation_length is None:
            max_tokens = lm.max_gen_toks
        else:
            max_tokens = max_generation_length

        token_context = lm.tok_encode_batch(context)

        responses = lm._model_generate(
            inputs=token_context,
            max_tokens=max_tokens,
            stop=until,
        )
        responses = lm.tok_decode(responses.tolist())

        results = []
        for response in responses:
            # Ensure the generated responses do not contain the stop sequences.
            for term in until:
                response = response.split(term)[0]
            results.append(response)

        return results[0]
    
    def generate_second_stage_input(self, stage1, generated_explanation):
        prompt_suffix = "Therefore, among A through E, the answer is"
        return stage1 + generated_explanation + "\n" + prompt_suffix

    def doc_to_fewshot_prompt(self, doc, num_fewshot):
        description = "Here are the answers for the problems in the exam.\n"

        # generate labeled examples
        def doc_to_question_input(doc, question_idx, with_explanation=True):
            return "Problem {}.   ".format(question_idx) + passage + " " + doc["question"] + "\n" + "Choose from the following options:    " + " ".join(doc["options"]) + "\n" \
            + ("Explanation for Problem {}:   ".format(question_idx) + doc['explanation'] + "\n" if with_explanation else "")\
            + "The anwser is therefore"
        
        labeled_examples = ""
        fewshotex = self.fewshot_examples(k=num_fewshot)
        for fewshot_idx, fewshot_doc in enumerate(fewshotex):
            passage = "" if fewshot_doc['doc']["passage"] is None else fewshot_doc['doc']["passage"]
            question_input = doc_to_question_input(fewshot_doc['doc'], fewshot_idx+1)
            question_output = self.doc_to_target(fewshot_doc)
            labeled_examples += question_input + question_output + "\n"

        end_of_labeled_example = "\n"

        example = doc_to_question_input(doc, num_fewshot+1, False)

        prompt = description + labeled_examples + end_of_labeled_example + example
        return prompt

    def doc_to_text(self, doc, num_fewshot, lm):
        query_prompt = ""
        if num_fewshot == 0:
            query_prompt = self.doc_to_zeroshot_prompt(doc["doc"], lm)
        elif num_fewshot > 0:
            query_prompt = self.doc_to_fewshot_prompt(doc["doc"], num_fewshot)

        # The query prompt portion of the document example.
        return query_prompt
    
    def fewshot_context(
        self, doc, num_fewshot, lm, provide_description=None, rnd=None, description=None
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
        return self.doc_to_text(doc, num_fewshot, lm)
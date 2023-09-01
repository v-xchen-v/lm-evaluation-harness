import os
import openai
import re
import logging
import time

openai.api_type = "azure"
openai.api_base = "https://agi-france-central.openai.azure.com/"
openai.api_version = "2023-03-15-preview"
openai.api_key = os.environ["OPENAI_API_SECRET_KEY"]

def _prompt_to_chatml(prompt: str, start_token: str = "<|im_start|>", end_token: str = "<|im_end|>"):
    r"""Convert a text prompt to ChatML formal

    Examples
    --------
    >>> prompt = (
    ... "<|im_start|>system\n"
    ... "You are a helpful assistant.\n<|im_end|>\n"
    ... "<|im_start|>system name=example_user\nKnock knock.\n<|im_end|>\n<|im_start|>system name=example_assistant\n"
    ... "Who's there?\n<|im_end|>\n<|im_start|>user\nOrange.\n<|im_end|>"
    ... )
    >>> print(prompt)
    <|im_start|>system
    You are a helpful assistant.
    <|im_end|>
    <|im_start|>system name=example_user
    Knock knock.
    <|im_end|>
    <|im_start|>system name=example_assistant
    Who's there?
    <|im_end|>
    <|im_start|>user
    Orange.
    <|im_end|>
    >>> _prompt_to_chatml(prompt)
    [{'content': 'You are a helpful assistant.', 'role': 'system'},
      {'content': 'Knock knock.', 'role': 'system', 'name': 'example_user'},
      {'content': "Who's there?", 'role': 'system', 'name': 'example_assistant'},
      {'content': 'Orange.', 'role': 'user'}]
    """
    prompt = prompt.strip()
    assert prompt.startswith(start_token)
    assert prompt.endswith(end_token)

    message = []
    for p in prompt.split("<|im_start|>")[1:]:
        newline_splitted = p.split("\n", 1)
        role = newline_splitted[0].strip()
        content = newline_splitted[1].split(end_token, 1)[0].strip()

        if role.startswith("system") and role != "system":
            # based on https://github.com/openai/openai-cookbook/blob/main/examples
            # /How_to_format_inputs_to_ChatGPT_models.ipynb
            # and https://github.com/openai/openai-python/blob/main/chatml.md it seems that system can specify a
            # dictionary of other args
            other_params = _string_to_dict(role.split("system", 1)[-1])
            role = "system"
        else:
            other_params = dict()

        message.append(dict(content=content, role=role, **other_params))

    return message

def gpt_4_completion(template_file_path, temperature=0.7, max_tokens=800, top_p=0.95, **kwargs):
    if not os.path.exists(template_file_path):
        raise ValueError(f"Template file does not exist: {template_file_path}")
  
    with open(template_file_path) as f:
        template = str.join("", f.readlines())

    # logging.info({"temperature": temperature, "max_tokens": max_tokens, "top_p": top_p})
  
    text_to_format = re.findall("{([^ \s]+?)}", template)
    for key in text_to_format:
        if key not in kwargs:
            raise ValueError(f"Missing keyword argument: {key}")
        template = template.replace("{" + key + "}", '\n'.join(kwargs[key]) if type(kwargs[key]) is list else kwargs[key])

    prompt = _prompt_to_chatml(str.join('', template))

    while True:
        try:
            response = openai.ChatCompletion.create(
                engine="gpt-4-deployment",
                messages = prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None)
            break
        except openai.error.OpenAIError as e:
            logging.warning(f"OpenAIError: {e}.")
            
            # dead loop when trigger 'The response was filtered due to the prompt triggering Azure OpenAI’s content management policy. Please modify your prompt and retry. To learn more about our content filtering policies please read our documentation: https://go.microsoft.com/fwlink/?linkid=2198766.'
            if str(e).startswith("The response was filtered due to the prompt triggering Azure OpenAI’s content management policy. Please modify your prompt and retry."):
                return {"response": str(e)}
            else:
                time.sleep(2)
  
    return {"response": response.choices[0].message.content, "total_tokens": response.usage.total_tokens}

if __name__ == '__main__':
    # res = gpt_4_completion("prompt.txt", question="What is the meaning of life?", answer="42", choices="A. 42\nB. 43\nC. 44\nD. 45")
    res = gpt_4_completion("prompt.txt", question="What is the meaning of life?", answer="42", choices=["A. 42","B. 43","C. 44","D. 45"])
    print(res)
    
    query_1 = 'Personal Care and Style: How to increase breast size with a bra. Check your bra size. Wearing a bra that is too big will not make your breasts look larger. That is why it is important to wear the right size bra for you.'
    choices_1 = ['You can visit a lingerie shop and have them measure you to help you fit a bra to your size, or measure yourself before you shop for a new bra to ensure that you get a good fit. Use a flexible tape measure, like one found in a sewing kit.', 'This is why it is important to keep your breasts under protection when in the shower and only wear bras that are larger than your breast size. If you are not wearing a bra, try wearing something that is a little bigger.', 'For a girl, a bra with a support strap will be easier for her, because most women are unable to pull through bra straps and bras that are too small will not be able to support breasts from side-to-side. Many bras have even been created that cover the breast side, and can be sent to other women in the world to make them look bigger.', 'Choose a color that is flattering to your breast type and specific event, in addition to those that make you uncomfortable. Look for sports bras made from natural material, such as spandex or lycra, as this is a more breathable bra.']
    answer_1 = '\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n'
    gold_1=0 # A
    res = gpt_4_completion("prompt.txt", question=query_1, answer=answer_1, choices=choices_1)
    print(res) # A, reansonable?
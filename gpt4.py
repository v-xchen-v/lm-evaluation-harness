import os
import openai
import re
import logging
import time

openai.api_type = "azure"
openai.api_base = "https://slrt-east-us.openai.azure.com/"
openai.api_version = "2023-07-01-preview"
openai.api_key = "14edc332b32f486baf542a90ba521ab5" 

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
        template = template.replace("{" + key + "}", kwargs[key])

    prompt = _prompt_to_chatml(str.join('', template))

    is_valid_response = True
    response_content = ""
    usage = None

    while True:
        try:
            response = openai.ChatCompletion.create(
                engine="gpt-4-0613",
                messages=prompt,
                temperature=0)
            response_content = response.choices[0].message.content
            usage = response.usage
            break
        # https://github.com/openai/openai-python/blob/main/openai/error.py
        except openai.error.RateLimitError as e:
            logging.warning(f"RateLimitError: {e}.")
            time.sleep(2)
        except openai.error.InvalidRequestError as e:
            logging.warning(f"InvalidRequestError: {e}.")
            if e.error.code == "content_filter":
                is_valid_response = False
                break
        except openai.error.TryAgain or openai.error.Timeout as e:
            logging.warning(f"TryAgain/Timeout: {e}.")
            time.sleep(2)
        except Exception as e:
            logging.warning(f"{e}")
            time.sleep(2)

  
    return {
        "is_valid": is_valid_response,
        "response": response_content,
        "usage": usage
    }

if __name__ == '__main__':
    # res = gpt_4_completion("prompt.txt", question="What is the meaning of life?", model1_answer="42", model2_answer="45")
    # print(res)
    # text = res['response']
    import re
    
    # Compile the regular expression
    pattern = re.compile(r"- Score for Model 1:\s*([0-9.]+)")

    # Search the text
    text = '- Score for Model 1: 0\n- Score for Model 2: 0\n\nBrief Explanation: \nBoth models failed to provide a complete and accurate answer to the question. Model 1 started to provide an answer but did not finish, while Model 2 did not provide any answer at all. Therefore, both models receive a score of 0.'
    # text = "Model 1's Answer: The meaning of life is that we are living in a - Score: 1\nModel 2's Answer: (No response) - Score: 0\n\nJustification: Model 1, despite providing an incomplete answer, at least attempted to answer the question. Model 2 did not provide any response, therefore it scores lower."
    match = pattern.search(text)

    # If a match is found, print the score
    if match:
        score = match.group(1)
        print(f"Model 1's Score is: {score}")
    else:
        print("No match found.")
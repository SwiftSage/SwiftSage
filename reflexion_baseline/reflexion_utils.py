from typing import List, Dict, Any, Optional, Union
# from tenacity import (
#     retry,
#     stop_after_attempt,
#     wait_random_exponential,
# )
from slow_agent.utils import completion_with_backoff

with open("./reflexion_baseline/reflexion_few_shot_examples.txt", 'r') as f:
    FEW_SHOT_EXAMPLES = f.read()

import os
import openai
import tiktoken
openai.api_key = os.environ["OPENAI_API_KEY"]

def _generate_reflection_query(prompt: str, memory: List[str], model_name: str = None) -> str:
    """Allows the Agent to reflect upon a past experience."""

    encoding = tiktoken.encoding_for_model(model_name)
    if model_name == "gpt-3.5-turbo":
        max_len = 4096
    elif model_name == "gpt-4":
        max_len = 8192
    else:
        max_len = 4097

    while True:
        query: str = f"""You will be given the history of a past experience in which you were placed in an environment and given a task to complete. You were unsuccessful in completing the task. Do not summarize your environment, but rather think about the strategy and path you took to attempt to complete the task. Devise a concise, new plan of action that accounts for your mistake with reference to specific actions that you should have taken. For example, if you tried A and B but forgot C, then devise a plan to achieve C with environment-specific actions. You will need this later when you are solving the same task. Give your plan after "Plan". Here are two examples:

{FEW_SHOT_EXAMPLES}

{prompt}STATUS: FAIL
"""

        if len(memory) > 0:
            query += '\n\nPlans from past attempts:\n'
            for i, m in enumerate(memory):
                query += f'Trial #{i}: {m}\n'

        query += '\n\nNew plan:'

        if len(encoding.encode(query)) > max_len - 266:
            index1 = prompt.find('>')
            index2 = prompt.find('>', index1+1)
            prompt = prompt[:index1] + prompt[index2:]
        else:
            break

    return query

def update_memory(prompt, env_configs, model_name):
    """Updates the given env_config with the appropriate reflections."""
    # if unsolved, get reflection and update env config
    if not env_configs['is_success']:
        reflection_query: str = _generate_reflection_query(prompt.split("Here is the task:")[-1].strip().strip('>'), env_configs['memory'][-3:], model_name=model_name)
        reflection: str = get_completion(reflection_query, model_name=model_name) # type: ignore
        env_configs['memory'] += [reflection]
                
    return env_configs

# @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_completion(prompt: Union[str, List[str]], max_tokens: int = 256, stop_strs: Optional[List[str]] = None, is_batched: bool = False, model_name: str = None) -> Union[str, List[str]]:
    assert (not is_batched and isinstance(prompt, str)) or (is_batched and isinstance(prompt, list))
    response = completion_with_backoff(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
    # response = openai.Completion.create(
        # model='text-davinci-003',
        # prompt=prompt,
        temperature=0.0,
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=stop_strs,
    )
    # if is_batched:
    #     res: List[str] = [""] * len(prompt)
    #     for choice in response.choices:
    #         res[choice.index] = choice.text
    #     return res
    return response["choices"][0]["message"]["content"]
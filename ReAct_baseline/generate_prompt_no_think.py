import json
from scienceworld import ScienceWorldEnv
import re

def clean(s):
    clean_toks = ['\n', '\t']
    for tok in clean_toks:
        s = s.replace(tok, ' ')
    return s

out = {}

for task_id in range(30):
    with open(f"traj_data_1shot/task_{task_id}.json", 'r') as f:
        all_data = json.load(f)
    
    for i in all_data.keys():
        if i == "task_id":
            pass

        elif i == "taskName":
            taskName = all_data[i]
        
        else:
            example_data = all_data[i]

    prompt = ""
    for step_key, step_data in example_data.items():

        if step_key == "task_description":
            task_description = step_data
            continue

        # Add first observation and task description
        elif step_key == "0":
            prompt += clean(step_data["observation"])
            prompt += "\n"
            prompt += task_description
        
        else:
            prompt += "\n"
            prompt += "> " +  step_data["action"]
            prompt += "\n"
            prompt += clean(step_data["observation"])
        
    out[task_id] = prompt
    prompt = ""


with open("prompt_no_think.jsonl", 'w') as f:
    f.write(json.dumps(out, indent=4))





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

        # Add first observation, task description and and first thinking
        elif step_key == "0":
            prompt += clean(step_data["observation"])
            prompt += "\n"
            prompt += task_description
            prompt += "\n"
            prompt += "> think: To solve the task, I need to " +  " and ".join(step_data["next_subgoal"])
            prompt += "\n"
            prompt += "OK."
        
        else:
            prompt += "\n"
            prompt += "> " +  step_data["action"]
            prompt += "\n"
            prompt += clean(step_data["observation"])
        
            # Add thinking action if current action is key action
            if step_data["is_key_action"]:

                # Add make sure before substance
                for idx, condition in enumerate(step_data["next_subgoal"]):
                    if condition[:9] == "substance":
                        step_data["next_subgoal"][idx] = "make sure " + condition
                
                for idx, condition in enumerate(step_data["completed_subgoals"]):
                    if condition[:9] == "substance":
                        step_data["completed_subgoals"][idx] = "make sure " + condition

                prompt += "\n"
                prompt += "> think: Now I "+ " and ".join(step_data["completed_subgoals"]).replace("be", "am")
                prompt += ". Next, I need to " +  " and ".join(step_data["next_subgoal"])
                prompt += "\n"
                prompt += "OK."

        
    out[task_id] = prompt
    prompt = ""


with open("prompt.jsonl", 'w') as f:
    f.write(json.dumps(out, indent=4))





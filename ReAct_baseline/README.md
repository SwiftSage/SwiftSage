
## Generate Prompt

copy traj_data_1shot from slow_agent to this directory, then generate prompts by running

```bash 
python generate_prompt.py
```
This prompt.jsonl will be used in ReAct and reflexion baseline

---

Generate prompts without "think" action by running
```bash 
python generate_prompt_no_think.py
```
This prompt_no_think.jsonl will be used in SayCan baseline
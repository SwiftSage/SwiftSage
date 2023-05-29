import json 
from scienceworld import ScienceWorldEnv
import os 

env = ScienceWorldEnv("", "", envStepLimit = 300)
taskNames = env.getTaskNames()

gold_data_path = "data_utils/goldsequences-0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20-21-22-23-24-25-26-27-28-29.json"
# for gold in golds:
with open(gold_data_path, 'r') as f:
    raw_data = json.load(f)

for _, curr_task in raw_data.items():
    taskName = curr_task["taskName"]
    if taskName.startswith("task"):  
        second_index = taskName.index('-', taskName.index('-') + 1)
        taskName = taskName[second_index+1:]
        taskName = taskName.replace("(","")
        taskName = taskName.replace(")","")  
    taskName = taskName.replace("mendellian", "mendelian")
    # assert taskName in taskNames, taskName 
    task_id = taskNames.index(taskName) 
    all_samples = curr_task['goldActionSequences'] 
    test_samples = [s for s in all_samples if s["fold"] == "test"][:10]
    # print(task_id, taskName, len(test_samples), test_samples[0]['variationIdx'], test_samples[-1]['variationIdx'])
    start = test_samples[0]['variationIdx']
    end = test_samples[-1]['variationIdx']
 
    
    all_vars = {}
    for var in test_samples:
        var_id = var["variationIdx"]
        all_vars[str(var_id)] = {}
        all_vars[str(var_id)]["episodeIdx"] = var_id
        all_vars[str(var_id)]["history"] = var
        history = all_vars[str(var_id)]["history"]["path"]
        del all_vars[str(var_id)]["history"]["path"]
        all_vars[str(var_id)]["history"]["taskName"] = taskName
        all_vars[str(var_id)]["history"]["task_id"] = task_id
        all_vars[str(var_id)]["history"]["history"] = history
        all_vars[str(var_id)]["notes"] = {"mode": "oracle"}
    #     print(json.dumps(all_vars, indent=2))
    #     break 
    # break 
    with open(os.path.join("analysis/oracle_log", f"task{task_id}-{start}-{end}.json"), "w") as f:
        json.dump(all_vars, f, indent=2)
import json  
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
import sys
import random 

random.seed(42)
task = sys.argv[1]
# task = "medium"

easy_tasks = [3, 5, 6, 7, 8, 13, 17, 18, 19, 25]
medium_tasks = [2, 4, 12, 20, 26, 27, 28, 29]
hard_tasks = [0, 1, 9, 10, 11, 14, 15, 16, 21, 22, 23, 24]

max_type = {"easy":30, "medium":50, "hard":120}
# for i in range(30):
#     if i in easy_tasks:
#         max_type[str(i)] = 30
#     elif i in medium_tasks:
#         max_type[str(i)] = 50
#     else:
#         max_type[str(i)] = 150

def add_random_float(lst, a=0,b=1):
    result = []
    for item in lst:
        random_float = random.uniform(a, b)
        new_item = item + random_float
        result.append(new_item)
    return result

def analyze_efficiency(folder, model, task):
    data = []
    for file in os.listdir(folder):
        filename = os.path.join(folder, file)
        if not filename.endswith(".json"):
            continue
        task_id = file.split("-")[0].replace("task", "")
        if task_id != task:
            continue
        with open(filename) as f:
            samples = json.load(f) 
        cnt = 0 
        K = 3
        for var_id, sample in samples.items():
            scores = [int(float(h['score'])*100) for h in sample['history']['history'] if float(h['score'])>=0]
            while scores[-1] == scores[-2] == 100:
                scores.pop(-1)
            end = len(scores)
            if model == "Oracle" and scores[-1] != 100:
                print(task_id, var_id, "WARN")
            for i in range(len(scores)-1):
                if scores[i]>scores[i+1]:
                    end = i + 1
                    break 
            scores = scores[:end]
            scores = add_random_float(scores)
            times = list(range(1, len(scores)+1))
            times = add_random_float(times, a=0, b=0.5)
            data.append({"model": model, "var_id": int(var_id), "task": task_id, "time": times, "score": scores})
            assert len(data[-1]['time']) == len(data[-1]['score'])
            cnt += 1
            # print(var_id)
            if cnt >= K:
                break 
            
    return data 


data = []

if task == "easy":
    tasks = easy_tasks
elif task == "medium":
    tasks = medium_tasks
elif task == "hard":
    tasks = hard_tasks
    
for t in tasks:
    t = str(t)
    data += analyze_efficiency("analysis/oracle_log", model="Oracle", task=t)
    data += analyze_efficiency("fast_slow_logs/final_merge_gpt4", model="SwiftSage", task=t)
    data += analyze_efficiency("ReAct_logs/gpt-4", model="ReAct", task=t)
    # data += analyze_efficiency("reflexion_logs/gpt-4", model="Reflexion", task=t)
    # data += analyze_efficiency("saycan_logs/gpt-4", model="SayCan", task=t)


max_time_length = max([len(x["time"]) for x in data])

# max_time_length = 30

for d in data:
    K = len(d['time'])
    if K >= max_time_length:
        d['time'] = d['time'][:max_time_length+1]
        d['score'] = d['score'][:max_time_length+1]
    else:
        # last_score = d['score'][-1]
        # if last_score == 100:
        last_score = None  
        d['time'] += list(range(K+1, max_time_length+1))
        d['score'] += [last_score] * (max_time_length - K ) 
        assert len(d['time']) == len(d['score'])
         

sns.set_palette("husl")
plt.figure(figsize=(8, 8))
markers = ['o', 'v', 's', '4']
marker_size = 3  # Adjust the marker size as desired
models = sorted(list(set(d['model'] for d in data)))
linestyles = [':', ':', ':', ':']
model_colors = {
    # "SayCan": "yellow",
    "SwiftSage": "blue",
    "ReAct": "red",
    "Oracle": "gray"
}

# Define a color palette for the models
model_palette = sns.color_palette("husl", len(models))

for d in data:
    model_index = models.index(d['model'])
    df = pd.DataFrame(d)

    # Sort the data points based on 'time' before plotting
    df = df.sort_values('time')

    x = df['time']
    y = df['score']

    sns.lineplot(
        x=x,
        y=y,
        data=d,
        label=d['model'],
        marker=markers[model_index],
        linestyle=linestyles[model_index],
        markersize=marker_size,  # Set the marker size
        color=model_colors[d['model']]
    )
    plt.fill_between(x, y, alpha=0.05, color=model_colors[d['model']])

plt.xlim(1, min(max_time_length, max_type[task]))
plt.legend(title='Models', loc='upper left', borderaxespad=0)
# if task != "0":
plt.legend().remove()
plt.xlabel('')
plt.ylabel('')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
task = task[0].upper() + task[1:]
plt.title(f'{task} Tasks')
plt.savefig(f'analysis/{task}.pdf', format='pdf')

"""
python analysis/efficiency.py easy
python analysis/efficiency.py medium
python analysis/efficiency.py hard
"""
 
import json  
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import numpy as np 
import sys
import random 
from tqdm import trange

def get_real_task_id(task_name):
    task_name = task_name.replace("mendelian", "mendellian")
    task_table = {
        'boil': '1-1',
        'melt': '1-2',
        'freeze': '1-3',
        'change-the-state-of-matter-of': '1-4',
        'use-thermometer': '2-1',
        'measure-melting-point-known-substance': '2-2',
        'measure-melting-point-unknown-substance': '2-3',
        'power-component': '3-1',
        'power-component-renewable-vs-nonrenewable-energy': '3-2',
        'test-conductivity': '3-3',
        'test-conductivity-of-unknown-substances': '3-4',
        'find-living-thing': '4-1',
        'find-non-living-thing': '4-2',
        'find-plant': '4-3',
        'find-animal': '4-4',
        'grow-plant': '5-1',
        'grow-fruit': '5-2',
        'chemistry-mix': '6-1',
        'chemistry-mix-paint-secondary-color': '6-2',
        'chemistry-mix-paint-tertiary-color': '6-3',
        'lifespan-longest-lived': '7-1',
        'lifespan-shortest-lived': '7-2',
        'lifespan-longest-lived-then-shortest-lived': '7-3',
        'identify-life-stages-1': '8-1',
        'identify-life-stages-2': '8-2',
        'inclined-plane-determine-angle': '9-1',
        'inclined-plane-friction-named-surfaces': '9-2',
        'inclined-plane-friction-unnamed-surfaces': '9-3',
        'mendellian-genetics-known-plant': '10-1',
        'mendellian-genetics-unknown-plant': '10-2'
    }
    return task_table.get(task_name)

random.seed(42)


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
        all_scores = []
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
            all_scores.append(scores)
            cnt += 1
            if cnt >= K:
                break 

        max_len = max([len(i) for i in all_scores])
        for scores in all_scores:
            scores += [scores[-1]] * (max_len - len(scores))
        
        times = list(range(1, max_len+1))
        times = add_random_float(times, a=0, b=0.5)

        all_scores = np.array(all_scores)
        all_scores = list(np.mean(all_scores, 0))
        all_scores = add_random_float(all_scores)
        data.append({"model": model, "var_id": int(var_id), "task": task_id, "time": times, "score": all_scores})
            
    return data 

tasks = [17, 19, 18, 7, 8, 25, 6, 5, 3, 13, 26, 29, 4, 27, 28, 2, 20, 12, 21, 11, 1, 22, 10, 15, 9, 14, 0, 16, 23, 24]
task_type = ['S'] * 10 + ['M'] * 8 + ['L'] * 12
fig, axes = plt.subplots(5, 6, figsize=(8, 4))
for i in trange(30):
    t = tasks[i]
    data = []
    t = str(t)
    task_score_file_name = f'reflexion_logs/gpt-3.5-turbo/task{t}-score.txt'

    with open(task_score_file_name) as f:
        text = f.readlines()
        index2 = text[1].find('Scores')
        task_name = text[1][10:index2]

    real_task_id = get_real_task_id(task_name) if task_name!="-" else "N/A"

    data += analyze_efficiency("analysis/oracle_log", model="Oracle", task=t)
    data += analyze_efficiency("fast_slow_logs/final_merge_gpt4", model="SwiftSage", task=t)
    data += analyze_efficiency("ReAct_logs/gpt-4", model="ReAct", task=t)
    # data += analyze_efficiency("reflexion_logs/gpt-4", model="Reflexion", task=t)
    # data += analyze_efficiency("saycan_logs/gpt-4", model="SayCan", task=t)

    max_time_length = max([len(x["time"]) for x in data])    

    sns.set_palette("husl")
    ax = axes.flatten()[int(i)]
    models = sorted(list(set(d['model'] for d in data)))
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
            color=model_colors[d['model']],
            ax=ax
        )

    if real_task_id == "4-2":
        max_time_length = max_time_length / 8
    
    if real_task_id == "3-1":
        max_time_length = max_time_length / 3

    if real_task_id == "6-2":
        max_time_length = max_time_length / 6
    
    if real_task_id == "8-2":
        max_time_length = max_time_length / 1.5
    
    if real_task_id == "3-2":
        max_time_length = max_time_length / 3
    
    if real_task_id == "2-1":
        max_time_length = max_time_length / 4
    
    if real_task_id == "6-3":
        max_time_length = max_time_length / 5

    if real_task_id == "3-3":
        max_time_length = max_time_length / 5 
    
    if real_task_id == "5-1":
        max_time_length = max_time_length / 2


    ax.set_xlim(1, max_time_length*1.03)
    ax.set_ylim(0, 105)
    ax.legend().remove()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_title(f'{real_task_id} {task_type[i]} ', fontsize=8, loc="right", y=-0.1)
fig.tight_layout()
fig.savefig(f'analysis/test_ave.png', format='png')
fig.savefig(f'analysis/test_ave.pdf', format='pdf')
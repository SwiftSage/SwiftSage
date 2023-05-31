import json 
import glob 
import numpy as np
from tabulate import tabulate
from data_utils.data_utils import get_real_task_id
import sys
from tqdm import tqdm
import tiktoken
encoding = tiktoken.encoding_for_model('gpt-4')

def get_result(file_name):
    results = {}
    log_files = glob.glob(file_name)
    for file_name in tqdm(log_files):
        task_id = file_name[:-4]
        task_score_file_name = task_id + '-score.txt'
        with open(task_score_file_name) as f:
            text = f.readlines()
            index2 = text[1].find('Scores')
            task_name = text[1][10:index2]

        with open(file_name) as f:
            text = f.readlines()
            n = len(text)
            task_len = []
            token_len = []

            length = 0
            flag = 0
            for i in range(n):
                if 'Prompt: Interact with a household to solve a task. Here is an example.' in text[i]:
                    flag = 1
                    index = text[i].find('Prompt')
                    length = len(encoding.encode(text[i][index:]))
                    # print(text[i])
                
                elif '[INFO' in text[i] and flag:
                    token_len.append(length)
                    flag = 0
                    length = 0
                    # print(text[i])
                    # exit()
                
                else:
                    length += len(encoding.encode(text[i]))
                    # print(text[i])
                
                # print(length)
                
            results[task_name] = {'ave_token': sum(token_len)/len(token_len), 'num_actions': len(token_len)}
            token_len = []
    
    return results

results_all = {}
results_all["react"] = get_result('ReAct_logs/gpt-4/*.log')
results_all["reflexion"] = get_result('reflexion_logs/gpt-4/*.log')
results_all["saycan"] = get_result('saycan_logs/gpt-4/*.log')
# print(results_all)
# exit()
cols = ["trid", "tid", "task_name", "saycan", "react", "reflexion"]
rows = []
rows_v2 = []
for i in range(30):

    task_score_file_name = f'reflexion_logs/gpt-3.5-turbo/task{i}-score.txt'

    with open(task_score_file_name) as f:
        text = f.readlines()
        index2 = text[1].find('Scores')
        task_name = text[1][10:index2]

    # task_name = taskid_to_name.get(str(i), "-")
    real_task_id = get_real_task_id(task_name) if task_name!="-" else "N/A"
    row = [real_task_id, i, task_name]
    for c in cols[3:]:
        # score = results_all[c][task_name]['ave_token']
        score = results_all[c][task_name]['num_actions']
        row.append("{:.2f}".format(score))

    rows.append(row)

avg_scores = []


for j in range(len(cols)-3):
    scores = [float(r[3+j]) for r in rows]
    avg_s = np.mean([s for s in scores])
    avg_scores.append("{:.2f}".format(avg_s))

rows.append(["-"*5]*2 + ["-"*5]*len(rows[0]))
rows.append(["-", "-" , "all tasks (avg)"] + avg_scores)
print(tabulate(rows, headers=cols, tablefmt="pipe", numalign="center"))

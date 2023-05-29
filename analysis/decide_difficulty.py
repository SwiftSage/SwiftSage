import json
import os 
import numpy as np 

folder = "analysis/oracle_log"

id2id = """
1-1     0
1-2     22
1-3     9
1-4     1
2-1     29
2-2     20
2-3     21
3-1     25
3-2     26
3-3     27
3-4     28
4-1     6
4-2     7
4-3     8
4-4     5
5-1     11
5-2     10
6-1     2
6-2     3
6-3     4
7-1     17
7-2     19
7-3     18
8-1     12
8-2     13
9-1     14
9-2     15
9-3     16
10-1    23
10-2    24
"""
id_order = []
for line in id2id.splitlines():
    if '    ' not in line:
        continue 
    id_order.append(int(line.split("    ")[1]))
print(id_order)
    

easy_tasks = []
medium_tasks = []
hard_tasks = []


alen_dict = {}
for file in os.listdir(folder):
    with open(os.path.join(folder, file)) as f:
        data = json.load(f)
    var_lens = [] 
    task_id = None
    task_name = None  
    for var, var_data in data.items():
        var_lens.append(len(var_data['history']['history']))
        task_id = var_data['history']['task_id']
        task_name = var_data['history']['taskName']
    alen = np.mean(var_lens)
    print(task_id, alen) 
    alen_dict[task_id] = alen
    # task_id = task_name
    if alen < 20:
        easy_tasks.append(task_id)
    elif alen < 50:
        medium_tasks.append(task_id)
    else:
        hard_tasks.append(task_id)
print(f"easy_tasks = {easy_tasks}")
print(f"medium_tasks = {medium_tasks}")
print(f"hard_tasks = {hard_tasks}")

for tid in id_order:
    print(tid, alen_dict[tid])
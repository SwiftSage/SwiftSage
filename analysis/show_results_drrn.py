import json 
import glob 
import numpy as np
from tabulate import tabulate
from data_utils.data_utils import get_real_task_id
import sys

taskid_to_name = {}

def get_analysis(r, cut_off=None):
    scores = []
    scores_v2 = []
    taskName = ""
    for var, data in r.items():
        history = data["history"]["history"]
        score = history[-1]["score"]
        score_v2 = score
        if score == "-1.0":
            score = history[-2]["score"]
            score_v2 = 0.0
        scores.append(float(score))
        scores_v2.append(float(score_v2))

        taskNameIndex = data["history"]["taskName"].index("-", 5)
        taskName = data["history"]["taskName"][taskNameIndex+1:].replace("(", "").replace(")", "")
    # if taskName == "mendelian-genetics-unknown-plant":
    #     scores = scores[:30]
    #     # print(len(scores))
    #     pass
    # scores = scores[-3:]
    # scores_v2 = scores_v2[-3:]
    avg_score = np.mean(scores)
    avg_score_v2 = np.mean(scores_v2)
    return avg_score, avg_score_v2, taskName

def add_reuslts(log_folder,results, model_names, model_name=None):
    json_files = glob.glob(log_folder + "/*.json")
    for jf in json_files:
        with open(jf) as f:
            r = json.load(f)
        task_start = jf.index("task")
        if "seed" in jf:
            task_end = jf.index("-seed")
        else:
            task_end = jf.index("-", task_start)
            # print(jf)
        task_num = jf[task_start+4:task_end-8]
        # print(task_num)
        if task_num not in results:
            results[task_num] = {}
        if model_name is None:
            model_name = log_folder
        if model_name not in model_names:
            model_names.append(model_name)
            # if "mini" not in model_name:
            #     model_names.append(model_name+"(mini)")
                # pass
        avg_score, avg_score_v2, taskName = get_analysis(r)
        results[task_num][model_name] = avg_score, avg_score_v2
        # if "mini" not in model_name:
        #     results[task_num][model_name+"(mini)"] = avg_score_mini, avg_score_mini_v2

        
        # taskName
        taskid_to_name[task_num] = taskName

results = {}
model_names = []


add_reuslts("drrn-scienceworld/eval_logs", results, model_names, "drrn-mini")


cols = ["trid", "old_tid", "task_name"] + model_names 
rows = []
rows_v2 = []
for i in range(30):
    task_name = taskid_to_name.get(str(i), "-")
    real_task_id = get_real_task_id(task_name) if task_name!="-" else "N/A"
    row = [real_task_id, i, task_name]
    row_v2 = row[:]
    for c in cols[3:]:
        try:
            score, score_v2 = results[str(i)][c]
            row.append("{:.2f}".format(score*100))
            # row_v2.append("{:.2f}".format(score_v2*100))
        except:
            row.append("-0.0001")
            # row_v2.append("-0.0001")
    row.append(max([float(s) for s in row[3:]]))
    rows.append(row)
    rows_v2.append(row_v2)

cols += ["max"]
cols = [c.replace("test_fl-", "").replace("-bm=5", "") for c in cols]
avg_scores = []
std_scores = []
avg_scores_v2 = []


# remove those rows with -0.0001 
if len(sys.argv) > 1 and sys.argv[1] == "--selected":
    rows = [r for r in rows if "-0.0001" not in r]


for j in range(len(cols)-3):
    scores = [float(r[3+j]) for r in rows]
    avg_s = np.mean([s for s in scores])
    avg_scores.append("{:.2f}".format(avg_s))
    # std_scores.append(np.std(scores))
    # scores_v2 = [float(r[3+j]) for r in rows_v2]
    # avg_scores_v2.append(np.mean(scores_v2))

 
# rows.sort(key=lambda x: "0"+x[0] if x[0].index("-")<=1 else x[0])

rows.append(["-"*5]*2 + ["-"*5]*len(rows[0]))
rows.append(["-", "-" , "all tasks (avg)"] + avg_scores)

cols = [c.replace("logs/","") for c in cols]
# rows.append(["-", "-" , "all tasks (std)"] + std_scores)
# rows.append(["-", "-" , "all tasks (stop)"] + avg_scores_v2)
print(tabulate(rows, headers=cols, tablefmt="pipe", numalign="center"))

print(len(rows)-2)

"""
| trid   | old_tid   | task_name                                        | drrn-mini   | max   |
|:-------|:----------|:-------------------------------------------------|:------------|:------|
| 1-1    | 0         | boil                                             | 1.67        | 1.67  |
| 1-4    | 1         | change-the-state-of-matter-of                    | 0.00        | 0.0   |
| 1-3    | 2         | freeze                                           | 0.00        | 0.0   |
| 1-2    | 3         | melt                                             | 3.33        | 3.33  |
| 2-2    | 4         | measure-melting-point-known-substance            | 5.67        | 5.67  |
| 2-3    | 5         | measure-melting-point-unknown-substance          | 6.00        | 6.0   |
| 2-1    | 6         | use-thermometer                                  | 6.25        | 6.25  |
| 3-1    | 7         | power-component                                  | 11.11       | 11.11 |
| 3-2    | 8         | power-component-renewable-vs-nonrenewable-energy | 8.33        | 8.33  |
| 3-3    | 9         | test-conductivity                                | 7.94        | 7.94  |
| 3-4    | 10        | test-conductivity-of-unknown-substances          | 9.52        | 9.52  |
| 4-4    | 11        | find-animal                                      | 16.67       | 16.67 |
| 4-1    | 12        | find-living-thing                                | 13.89       | 13.89 |
| 4-2    | 13        | find-non-living-thing                            | 63.89       | 63.89 |
| 4-3    | 14        | find-plant                                       | 13.89       | 13.89 |
| 5-2    | 15        | grow-fruit                                       | 12.50       | 12.5  |
| 5-1    | 16        | grow-plant                                       | 7.06        | 7.06  |
| 6-1    | 17        | chemistry-mix                                    | 10.71       | 10.71 |
| 6-2    | 18        | chemistry-mix-paint-secondary-color              | 26.67       | 26.67 |
| 6-3    | 19        | chemistry-mix-paint-tertiary-color               | 8.89        | 8.89  |
| 7-1    | 20        | lifespan-longest-lived                           | 50.00       | 50.0  |
| 7-3    | 21        | lifespan-longest-lived-then-shortest-lived       | 33.33       | 33.33 |
| 7-2    | 22        | lifespan-shortest-lived                          | 50.00       | 50.0  |
| 8-1    | 23        | identify-life-stages-1                           | 20.00       | 20.0  |
| 8-2    | 24        | identify-life-stages-2                           | 20.00       | 20.0  |
| 9-1    | 25        | inclined-plane-determine-angle                   | 10.00       | 10.0  |
| 9-2    | 26        | inclined-plane-friction-named-surfaces           | 10.00       | 10.0  |
| 9-3    | 27        | inclined-plane-friction-unnamed-surfaces         | 10.00       | 10.0  |
| 10-1   | 28        | mendellian-genetics-known-plant                  | 17.00       | 17.0  |
| 10-2   | 29        | mendellian-genetics-unknown-plant                | 17.00       | 17.0  |
| -----  | -----     | -----                                            | -----       | ----- |
| -      | -         | all tasks (avg)                                  | 15.71       | 15.71 |
"""
import json 
import glob 
import numpy as np
from tabulate import tabulate
import sys 
import os
data_utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_utils'))
# Add the data_utils directory to the Python path
sys.path.append(data_utils_path)
from data_utils.data_utils import get_real_task_id

taskid_to_name = {}

def get_analysis(r, cut_off=10):
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
        taskName = data["history"]["taskName"]
    # if taskName == "mendelian-genetics-unknown-plant":
    #     scores = scores[:30]
    #     # print(len(scores))
    #     pass
    scores = scores[:cut_off]
    scores_v2 = scores_v2[:cut_off]
    avg_score = np.mean(scores)
    avg_score_v2 = np.mean(scores_v2)
    return avg_score, avg_score_v2, taskName

def add_reuslts(log_folder,results, model_names, model_name=None):
    json_files = glob.glob(log_folder + "/*.json")
    for jf in json_files:
        if jf.endswith("demos.json"):
            continue
        with open(jf) as f:
            r = json.load(f)
        task_start = jf.index("task")
        if "seed" in jf:
            task_end = jf.index("-seed")
        else:
            task_end = jf.index("-", task_start)
            # print(jf)
        task_num = jf[task_start+4:task_end]
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
        avg_score_mini, avg_score_mini_v2, _ = get_analysis(r, cut_off=3)
        results[task_num][model_name] = avg_score, avg_score_v2
        if "fast only" not in model_name:
            results[task_num][model_name+"(mini)"] = avg_score_mini, avg_score_mini_v2

        
        # taskName
        taskid_to_name[task_num] = taskName

results = {}
model_names = [] 

# add_reuslts("saycan_logs/gpt-3.5-turbo", results, model_names)
# add_reuslts("saycan_logs/gpt-4", results, model_names)


# add_reuslts("ReAct_logs/gpt-3.5-turbo", results, model_names)
# add_reuslts("ReAct_logs/gpt-4", results, model_names)

# add_reuslts("reflexion_logs/gpt-3.5-turbo", results, model_names)
# add_reuslts("reflexion_logs/gpt-4", results, model_names)


add_reuslts("logs/test_fl-v4.1-500-bm=5_nosbert",results, model_names, "Ours (Swift)")

# add_reuslts("fast_slow_logs/test_mini_all_v3_gpt-4",results, model_names, "Ours (v1)")

# add_reuslts("fast_slow_logs/test_mini_all_v5_gpt-4",results, model_names, "Ours (v2)")

# add_reuslts("fast_slow_logs/test_mini_all_0418_v1_gpt-4",results, model_names, "Ours (v3)") 

# add_reuslts("fast_slow_logs/test_mini_all_0419_v1_gpt-4",results, model_names, "Ours (v4)") 

# add_reuslts("fast_slow_logs/test_mini_all_0422_v1_gpt-3.5-turbo",results, model_names, "Ours (gpt-3.5)") 
# add_reuslts("fast_slow_logs/test_mini_all_0422_v1_gpt-4/",results, model_names, "Ours (final)") 

# add_reuslts("?fast_slow_logs/test_mini_all_0424_v1_gpt-3.5-turbo",results, model_names, "Ours (gpt-3.5 0424)") 
# add_reuslts("fast_slow_logs/test_mini_all_0425_v1_gpt-3.5-turbo",results, model_names, "Ours (gpt-3.5 0425)") 
# add_reuslts("fast_slow_logs/test_mini_all_0424_v1_gpt-4/",results, model_names, "Ours (final-0424)") 
# add_reuslts("fast_slow_logs/test_mini_all_0425_v1_gpt-4/",results, model_names, "Ours (final-0425)") 
# add_reuslts("fast_slow_logs/test_mini_2_all_0512_gpt-4/",results, model_names, "Ours (v2-0512)") 
# add_reuslts("fast_slow_logs/final_merge_gpt4",results, model_names, "Ours (merged-10)") 
add_reuslts("fast_slow_logs/final_merge_gpt-3.5-turbo/",results, model_names, "Ours (merged-10-3.5)") 

# add_reuslts("analysis/oracle_log",results, model_names, "Oracle") 







# add_reuslts("logs/test_fl-v4.1-800-bm=5_sbert",results, model_names)

########

# add_reuslts("logs/test_mini_fl-v3-300-bm=5_sbert",results, model_names)

# add_reuslts("logs/test_flan_large_v2-300-1337",results, model_names)
# add_reuslts("logs/test_flan_large_v2-300-bm10",results, model_names)
# add_reuslts("logs/test_flan_large_v2-300-bm5_fixobs",results, model_names)
# add_reuslts("logs/test_flan_large_v2-300-bm10_fixobs",results, model_names)
# add_reuslts("logs/test_flan_large_v2-350",results, model_names)
# add_reuslts("logs/test_flan_large_v2-400",results, model_names)

# add_reuslts("logs/test_mini_fl-v3-300-bm=5",results, model_names)
# add_reuslts("logs/test_mini_fl0403-bm10",results, model_names)
# add_reuslts("logs/test_mini_fl0404-200-bm=10",results, model_names)


# add_reuslts("logs/test_flan_large-300",results, model_names)
# add_reuslts("logs/test_flan_large-600",results, model_names)
# # add_reuslts("logs/test_flan_large-700",results, model_names)
# # add_reuslts("logs/test_flan_large-800",results, model_names)
# # add_reuslts("logs/test_flan_large-900",results, model_names)
# add_reuslts("logs/test_flan_large-1000",results, model_names)

# add_reuslts("logs/dev_t5l-v2-1200",results, model_names)
# add_reuslts("logs/dev_t5l-v2-1200_diverse",results, model_names)
# add_reuslts("logs/dev_flan_small",results, model_names)
# add_reuslts("logs/dev_fast_t5_base",results, model_names)
# add_reuslts("logs/dev_fast_t5_large",results, model_names)


# print(results)
cols = ["trid", "tid", "task_name"] + model_names 
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
            row.append("{:.1f}".format(score*100))
            row_v2.append("{:.1f}".format(score_v2*100))
        except:
            row.append("-0.0001")
            row_v2.append("-0.0001")
    # row.append(max([float(s) for s in row[-2:]]))
    rows.append(row)
    # row_v2.append(max([float(s) for s in row_v2[-2:]]))
    rows_v2.append(row_v2)

# cols += ["Ours(max)"]
cols = [c.replace("-turbo", "") for c in cols]
avg_scores = []
std_scores = []
avg_scores_v2 = []


# remove those rows with -0.0001 
if len(sys.argv) > 1 and sys.argv[1] == "--selected":
    rows = [r for r in rows if "-0.0001" not in r]


for j in range(len(cols)-3):
    scores = [float(r[3+j]) for r in rows]
    avg_s = np.mean([s for s in scores if s > 0])
    avg_scores.append("{:.2f}".format(avg_s))
    # std_scores.append(np.std(scores))
    scores_v2 = [float(r[3+j]) for r in rows_v2]
    avg_s = np.mean([s for s in scores_v2 if s >= 0])
    avg_scores_v2.append("{:.2f}".format(avg_s))

 
rows.sort(key=lambda x: "0"+x[0] if x[0].index("-")<=1 else x[0])

rows.append(["-"*5]*2 + ["-"*5]*len(rows[0]))
rows.append(["-", "-" , "all tasks (avg)"] + avg_scores)

cols = [c.replace("logs/","") for c in cols]
# rows.append(["-", "-" , "all tasks (std)"] + std_scores)
# rows.append(["-", "-" , "all tasks (avg, strict)"] + avg_scores_v2)
print()
# print(tabulate(rows, headers=cols, tablefmt="pipe", numalign="center"))



# for latex printing 

rows = [ [row[0]] + row[2:] for row in rows ]
# rows = [ r[:2] + [float(i) for i in r[2:]] for r in rows  ]
cols = [cols[0]] + cols[2:]
print(tabulate(rows, headers=cols, tablefmt="csv", numalign="center"))

print()
print(len(rows)-2)

 
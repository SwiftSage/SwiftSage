import json 
import glob 
import numpy as np
from tabulate import tabulate
from data_utils.data_utils import get_real_task_id
import sys
from tqdm import tqdm

log_files = glob.glob("reflexion_logs/gpt-4/*.log")
results = {}

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
        all_scores = []

        for i in range(n):
            if 'Run completed' in text[i]:
                j = i - 1
                flag = 1
                while flag:
                    if 'Scores' in text[j]:
                        index2 = text[j].find(']', 44)
                        scores = [int(float(j)) for j in text[j][44:index2].split(',')]
                        while len(scores) < 10:
                            scores.append(scores[-1])
                        flag = 0
                    j -= 1
               
                all_scores.append(scores)
                scores = []
        all_scores = np.array(all_scores)
        median_score = np.mean([np.median(i) for i in all_scores])

        score1= np.mean([i[0] for i in all_scores])
        score2 = np.mean([i[1] for i in all_scores])
        score3 = np.mean([i[2] for i in all_scores])
        score4 = np.mean([i[3] for i in all_scores])
        score5 = np.mean([i[4] for i in all_scores])

        last_score = np.mean([i[-1] for i in all_scores])
        results[task_name] = {'median': median_score, 'first': score1, 'second': score2, 'third': score3, 'fourth': score4, 'fifth': score5, 'last': last_score}
        all_scores = []

cols = ["trid", "tid", "task_name", "median", "first", "second", "third", "fourth", "fifth", "last"]
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
        try:
            score = results[task_name][c]
            row.append("{:.2f}".format(score))
        except:
            row.append("-0.0001")
    rows.append(row)

avg_scores = []


for j in range(len(cols)-3):
    scores = [float(r[3+j]) for r in rows]
    avg_s = np.mean([s for s in scores])
    avg_scores.append("{:.2f}".format(avg_s))

rows.append(["-"*5]*2 + ["-"*5]*len(rows[0]))
rows.append(["-", "-" , "all tasks (avg)"] + avg_scores)
print(tabulate(rows, headers=cols, tablefmt="pipe", numalign="center"))

# gpt-3.5-turbo
# | trid   | tid   | task_name                                        | median   | second   | last   |
# |:-------|:------|:-------------------------------------------------|:---------|:---------|:-------|
# | 1-1    | 0     | boil                                             | 0.00     | 0.67     | 1.00   |
# | 1-4    | 1     | change-the-state-of-matter-of                    | 0.00     | 1.00     | 0.00   |
# | 6-1    | 2     | chemistry-mix                                    | 30.50    | 26.33    | 51.33  |
# | 6-2    | 3     | chemistry-mix-paint-secondary-color              | 45.00    | 13.33    | 43.33  |
# | 6-3    | 4     | chemistry-mix-paint-tertiary-color               | 6.33     | 9.67     | 6.33   |
# | 4-4    | 5     | find-animal                                      | 36.00    | 5.67     | 36.00  |
# | 4-1    | 6     | find-living-thing                                | 43.00    | 16.67    | 41.67  |
# | 4-2    | 7     | find-non-living-thing                            | 90.33    | 83.33    | 72.33  |
# | 4-3    | 8     | find-plant                                       | 16.67    | 11.00    | 14.00  |
# | 1-3    | 9     | freeze                                           | 0.00     | 0.00     | 3.33   |
# | 5-2    | 10    | grow-fruit                                       | 45.50    | 19.00    | 72.67  |
# | 5-1    | 11    | grow-plant                                       | 8.67     | 7.67     | 5.67   |
# | 8-1    | 12    | identify-life-stages-1                           | 4.00     | 1.33     | 4.00   |
# | 8-2    | 13    | identify-life-stages-2                           | 8.00     | 5.33     | 6.67   |
# | 9-1    | 14    | inclined-plane-determine-angle                   | 35.83    | 35.00    | 80.00  |
# | 9-2    | 15    | inclined-plane-friction-named-surfaces           | 51.67    | 35.00    | 70.00  |
# | 9-3    | 16    | inclined-plane-friction-unnamed-surfaces         | 0.00     | 0.00     | 0.00   |
# | 7-1    | 17    | lifespan-longest-lived                           | 66.67    | 41.67    | 66.67  |
# | 7-3    | 18    | lifespan-longest-lived-then-shortest-lived       | 52.67    | 44.33    | 50.00  |
# | 7-2    | 19    | lifespan-shortest-lived                          | 50.00    | 66.67    | 50.00  |
# | 2-2    | 20    | measure-melting-point-known-substance            | 7.00     | 30.67    | 18.00  |
# | 2-3    | 21    | measure-melting-point-unknown-substance          | 3.33     | 0.33     | 3.67   |
# | 1-2    | 22    | melt                                             | 0.00     | 0.00     | 1.00   |
# | 10-1   | 23    | mendelian-genetics-known-plant                   | 19.83    | 0.00     | 33.67  |
# | 10-2   | 24    | mendelian-genetics-unknown-plant                 | 3.33     | 6.00     | 0.00   |
# | 3-1    | 25    | power-component                                  | 33.33    | 23.33    | 33.33  |
# | 3-2    | 26    | power-component-renewable-vs-nonrenewable-energy | 14.33    | 12.67    | 16.00  |
# | 3-3    | 27    | test-conductivity                                | 14.83    | 6.67     | 38.00  |
# | 3-4    | 28    | test-conductivity-of-unknown-substances          | 38.00    | 78.00    | 38.00  |
# | 2-1    | 29    | use-thermometer                                  | 5.50     | 2.00     | 8.00   |
# | -----  | ----- | -----                                            | -----    | -----    | -----  |
# | -      | -     | all tasks (avg)                                  | 24.34    | 19.44    | 28.82  |

# gpt-4
# | trid   | tid   | task_name                                        | median   | second   | last   |
# |:-------|:------|:-------------------------------------------------|:---------|:---------|:-------|
# | 1-1    | 0     | boil                                             | 3.33     | 5.00     | 3.33   |
# | 1-4    | 1     | change-the-state-of-matter-of                    | 23.50    | 0.00     | 24.00  |
# | 6-1    | 2     | chemistry-mix                                    | 77.67    | 80.67    | 77.67  |
# | 6-2    | 3     | chemistry-mix-paint-secondary-color              | 53.33    | 53.33    | 53.33  |
# | 6-3    | 4     | chemistry-mix-paint-tertiary-color               | 13.33    | 39.67    | 31.00  |
# | 4-4    | 5     | find-animal                                      | 100.00   | 47.33    | 100.00 |
# | 4-1    | 6     | find-living-thing                                | 47.33    | 19.67    | 72.33  |
# | 4-2    | 7     | find-non-living-thing                            | 100.00   | 100.00   | 100.00 |
# | 4-3    | 8     | find-plant                                       | 41.83    | 36.33    | 64.00  |
# | 1-3    | 9     | freeze                                           | 6.67     | 3.33     | 17.67  |
# | 5-2    | 10    | grow-fruit                                       | 15.00    | 17.33    | 15.00  |
# | 5-1    | 11    | grow-plant                                       | 7.17     | 7.33     | 7.33   |
# | 8-1    | 12    | identify-life-stages-1                           | 4.00     | 4.00     | 15.00  |
# | 8-2    | 13    | identify-life-stages-2                           | 8.00     | 8.00     | 12.00  |
# | 9-1    | 14    | inclined-plane-determine-angle                   | 100.00   | 83.33    | 100.00 |
# | 9-2    | 15    | inclined-plane-friction-named-surfaces           | 58.33    | 53.33    | 58.33  |
# | 9-3    | 16    | inclined-plane-friction-unnamed-surfaces         | 50.83    | 13.33    | 100.00 |
# | 7-1    | 17    | lifespan-longest-lived                           | 83.33    | 66.67    | 100.00 |
# | 7-3    | 18    | lifespan-longest-lived-then-shortest-lived       | 83.17    | 66.33    | 94.33  |
# | 7-2    | 19    | lifespan-shortest-lived                          | 83.33    | 83.33    | 100.00 |
# | 2-2    | 20    | measure-melting-point-known-substance            | 6.67     | 6.67     | 6.67   |
# | 2-3    | 21    | measure-melting-point-unknown-substance          | 6.00     | 6.00     | 6.00   |
# | 1-2    | 22    | melt                                             | 26.17    | 1.67     | 27.33  |
# | 10-1   | 23    | mendelian-genetics-known-plant                   | 41.83    | 33.67    | 100.00 |
# | 10-2   | 24    | mendelian-genetics-unknown-plant                 | 18.00    | 17.00    | 17.33  |
# | 3-1    | 25    | power-component                                  | 81.00    | 30.33    | 81.00  |
# | 3-2    | 26    | power-component-renewable-vs-nonrenewable-energy | 19.33    | 19.33    | 22.67  |
# | 3-3    | 27    | test-conductivity                                | 64.00    | 56.67    | 70.00  |
# | 3-4    | 28    | test-conductivity-of-unknown-substances          | 74.67    | 94.33    | 74.67  |
# | 2-1    | 29    | use-thermometer                                  | 9.50     | 7.00     | 14.67  |
# | -----  | ----- | -----                                            | -----    | -----    | -----  |
# | -      | -     | all tasks (avg)                                  | 43.58    | 35.37    | 52.19  |
import json 
import os 
from collections import defaultdict

test_mini_folder = "fast_slow_logs/test_mini_all_0425_v1_gpt-3.5-turbo"
test_mini_2_folder = "fast_slow_logs/test_mini_2_all_0512_gpt-3.5-turbo"
final_folder = "fast_slow_logs/final_merge_gpt-3.5-turbo/"


task_files = defaultdict(list)

for foldername in [test_mini_folder, test_mini_2_folder]:
    for f in sorted(os.listdir(foldername)):
        if f.endswith(".json") and f.startswith("task"):
            taskid = f.split("-")[0]
            task_files[taskid].append(os.path.join(foldername, f))
            
for task, files in task_files.items():
    data = {}
    start = 9999
    end = -1
    for file in files:
        _, s, e = file.split("/")[-1].replace(".json", "").split("-")
        start = min(start, int(s))
        end = max(end, int(e))
        with open(file) as f:
            data.update(json.load(f))
    # print(len(data))
    # print(start, end)
    assert len(data) == end-start+1
    with open(os.path.join(final_folder, f"{task}-{start}-{end}.json"), "w") as f:
        json.dump(data, f, indent=2)

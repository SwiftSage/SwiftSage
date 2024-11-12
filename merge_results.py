import os
import json


base_path = "output/math-l5/tmp_4a502776f1/math-l5"
shard_num = 8
res = []

for i in range(shard_num):
    with open(os.path.join(base_path, f"{i}", f"math-l5_score.jsonl"), "r") as fr:
        for line in fr:
            res.append(json.loads(line)['score'])

print(sum(res) / len(res))

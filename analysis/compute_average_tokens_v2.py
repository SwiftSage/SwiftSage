import json 
import glob 
import numpy as np
from tabulate import tabulate
# from data_utils.data_utils import get_real_task_id
import sys
from tqdm import tqdm
import tiktoken
encoding = tiktoken.encoding_for_model('gpt-4')

def get_result(file_name):
    results = {}
    log_files = glob.glob(file_name)
    for file_name in tqdm(log_files):
        task_id = file_name.split("/")[-1].replace("task", "").replace(".log", "")
        # task_score_file_name = task_id + '-score.txt'
        # with open(task_score_file_name) as f:
        #     text = f.readlines()
        #     index2 = text[1].find('Scores')
        #     task_name = text[1][10:index2]

        with open(file_name) as f:
            print(file_name)
            text = f.readlines()
            n = len(text)
            task_len = []
            token_len = []

            lenth = 0
            flag = 0
            num_actions = 0
            prompts = []
            for i in range(n):
                if '--prompt_to_plan--' in text[i] or '--prompt_to_next_actions--' in text[i]:
                    flag = 1
                    # index = text[i].find('You are an experienced teacher')
                    # lenth = len(encoding.encode(text[i][:]))
                    # print(text[i])
                elif '----------------------------------------------------------------------' in text[i] and flag:
                    token_len.append(lenth)
                    flag = 0
                    lenth = 0
                    # print(text[i])
                    # exit()
                if flag==1 and '[INFO' not in text[i]:
                    lenth += len(encoding.encode(text[i]))
                    # print(text[i])
                    prompts.append(text[i])
                
                if '[INFO	] Action:'  in text[i]:
                    num_actions+=1
                # print(lenth)
            # print(prompts)
            # print(num_actions) 
            results[task_id] = {'num_tokens': sum(token_len), 'num_actions': num_actions}
            token_len = []
    
    return results

results_all = {}
results_all["ss"] = get_result('fast_slow_logs/test_mini_all_0424_v1_gpt-4/task*.log')
print(results_all)
all_tokens = 0
all_actions = 0

for task_id in range(30):
    result = results_all['ss'][str(task_id)]
    all_tokens += result['num_tokens']  
    all_actions += result['num_actions']   
    print(f"{task_id}, {result['num_tokens']}, {result['num_actions']}")
print(all_tokens, all_actions, all_tokens/all_actions)
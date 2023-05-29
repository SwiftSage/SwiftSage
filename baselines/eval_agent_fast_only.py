
import argparse
import os
import re
import time
import torch
import random
import copy
from scienceworld import ScienceWorldEnv
import json
from tqdm import trange
from data_utils.data_utils import add_current_place, add_current_objects, sanitizeStr, formalize_action, recover_action
from data_utils.data_utils import compose_instance_v1, compose_instance_v1_1, compose_instance_v2, compose_instance_v3, compose_instance_v4, compose_instance_v5
from eval_utils import load_model, findValidActionNew, load_variation, get_model_output




import logging
from logging import INFO, WARN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_file_name(args, task_num):
    if (len(args["output_path"]) > 0):
        args["output_path"] += "/"

        # Make path if it doesn't exist
        if not os.path.exists(args['output_path']):
            os.makedirs(args["output_path"]) 
    # filenameOutPrefix = args["output_path"] + "transformer-" + args["mode"] + "-eval-" + str(args["lm_path"].split('/')[-1]) + "-task" + str(task_num)
    filenameOutPrefixSeed = args["output_path"] + "task" + str(task_num)

    return filenameOutPrefixSeed
  


# Example user input console, to play through a game.
def eval(args, task_num, logger):
    if args["compose_mode"] == "v1":
        compose_instance = compose_instance_v1
    elif args["compose_mode"] == "v1_1":
        compose_instance = compose_instance_v1_1
    elif args["compose_mode"] == "v2":
        compose_instance = compose_instance_v2
    elif args["compose_mode"] == "v3":
        compose_instance = compose_instance_v3
    elif args["compose_mode"] == "v4":
        compose_instance = compose_instance_v4
    elif args["compose_mode"] == "v5":
        compose_instance = compose_instance_v5
    

    # Initialize environment
    # env = ScienceWorldEnv("", args["jar_path"], envStepLimit = args["env_step_limit"], threadNum = 0)
    env = ScienceWorldEnv("", args["jar_path"], envStepLimit = args["env_step_limit"])
    taskNames = env.getTaskNames()
    taskName = taskNames[task_num]
    env.load(taskName, 0, args['simplification_str'])
    lm_model, tokenizer, sbert_model = load_model(args, device)
    variations = load_variation(env, args, task_num, logger)
    filenameOutPrefixSeed = get_file_name(args, task_num)
    # plans = get_plans(args)

    scores = []

    for variation in variations:

        # train_data = []
        env.load(taskName, variation, args["simplification_str"], generateGoldPath=True)
        task_description = env.taskdescription()[18:]
        # task_description = env.taskdescription()  
        recent_actions = ["look around"]
        recent_obs = ["N/A"]
        recent_scores = [0.0,]
        recent_reward = [0.0]
        # recent_actions_without_open = []
        places = []
        objects = [] 
        # bad_words_ids = None
 
        obs, info = env.reset()

        prev_obs = 'N/A'
        prev_action = 'look around'
        # prev_look = ''
        # prev_inv = ''

        done = False
        score = 0.0
        last_score = 0.0
        step = 0

        # The env has an internal step count, some actions like look around are free
        # however, the t5 model only generates the action "look around", which will result in a dead loop below
        # so the max_steps here is only used to avoid the model generating the same action forever
        max_steps = args["env_step_limit"] * 2
 
 
        while not done:
            input_str = ""

            add_current_place(obs, info['look'], places)
            add_current_objects(task_num, info['look'], objects, limit=20)


            # Note that the agent is allowed to know the score changes.
            returns_to_go = 1.0 - float(info['score']) * 0.01
            returns_to_go = round(returns_to_go, 2)
            

            mode = args["mode"]
            logger.info("Mode: " + mode)

            input_str, _ = compose_instance(mode=mode, step_id=step+1, task_desc=task_description, returns_to_go=returns_to_go,
                                    curr_action=None, curr_obs=obs, inventory=info['inv'], look=info['look'], 
                                    prev_action=prev_action, prev_obs=prev_obs, objects=objects, places=places, 
                                    recent_actions=recent_actions, recent_obs=recent_obs, recent_scores=recent_scores, recent_reward=recent_reward) 
            
            input_str = sanitizeStr(input_str)
            logger.info("InputStr: " + input_str)
            predStrs = get_model_output(args, input_str, tokenizer, lm_model, device, logger)
            
            ############
            prev_obs = obs 

            if args["compose_mode"] == "v5":
                predStrs = [recover_action(ps) for ps in predStrs]
                predStrs = [ps for ps in predStrs if ps]
            

            # Get valid actions at this point
            action = findValidActionNew(predStrs, env, info['look'], recent_actions, sbert_model, logger) 
            obs, reward, done, info = env.step(action)

            if args["compose_mode"] == "v5":
                action = formalize_action(action)

            score = info['score']
            prev_action = action
            reward = score - last_score
            recent_reward.append(reward/100)
            recent_scores.append(score/100)
            recent_actions.append(action) 
            recent_obs.append(obs)
            
            if score < 0:
                # Our own solution for dealing with such cases
                if args["no_stop"]:
                    done = True
                    score = last_score
                else:
                    done = True
                    score = 0
            last_score = score

            #logger.info("Input string: " + str(input_str))
            logger.info(f"Variation: {variation}, Step: {step}, Action: {action}")
            logger.info("Obs: " + sanitizeStr(obs))
            logger.info(f"Score: {score}")
            logger.info("")

            step += 1
            if (step >= max_steps) or done:
                break
  

            logger.info("Recent Actions: " + str(recent_actions))

            # Early stopping if we're in a loop
            if len(recent_actions) >= 5 and len(set(recent_actions[-5:])) == 2:
                logger.info("Many recent actions in history are the same -- model is likely in a loop, stopping early.")
                break


        # Store results
        env.storeRunHistory(variation, notes = {'mode':args["mode"], 'lm':str(args["lm_path"])} )
        env.saveRunHistoriesBufferIfFull(filenameOutPrefixSeed, maxPerFile=args["max_episode_per_file"])

        scores.append(score)

        logger.info("Run completed...")
        logger.info("Scores: " + str(scores))
 
        time.sleep(2)

    # Episodes are finished -- manually save any last histories still in the buffer
    env.saveRunHistoriesBufferIfFull(filenameOutPrefixSeed, maxPerFile=args["max_episode_per_file"], forceSave=True)

    avg = sum(scores) / len(scores)
    logger.info("Average score: " + str(avg))

    f = open(filenameOutPrefixSeed + "-score.txt", "a")
    f.write("\n" + "Task name:" + taskName + "Scores: " + str(scores) + " Average score: " + str(avg) + " Args: " + str(args) + "\n")
    f.close()

    logger.info("Shutting down server...")
    # env.shutdown()

    logger.info("Completed.")



def parse_args():
    parser = argparse.ArgumentParser()
    debug = False
    if not debug:
        parser.add_argument("--jar_path", type=str) 
        parser.add_argument("--task_nums", default="0")  # use comma to split 
        parser.add_argument("--env_step_limit", type=int, default=100)
        parser.add_argument("--lm_path", default="lm_model") 
        parser.add_argument("--simplification_str", default="easy")
        parser.add_argument("--beams", type=int, default=16)
        parser.add_argument("--max_episode_per_file", type=int, default=9999)
        parser.add_argument("--mode", default="fast_system")
        parser.add_argument("--set", default="test")
        parser.add_argument("--output_path", default="")
        parser.add_argument("--compose_mode", default="v1_1")
        parser.add_argument("--model_parallelism_size", type=int, default=1)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--max_input_len", type=int, default=1024)
        parser.add_argument("--cut_off", action="store_true", default=False)
        parser.add_argument("--no_stop", action="store_true", default=False)
        parser.add_argument("--sbert", action="store_true", default=False)
    else:
        parser.add_argument("--jar_path", type=str) 
        parser.add_argument("--task_nums", default="28")  # use comma to split 
        parser.add_argument("--env_step_limit", type=int, default=100)
        parser.add_argument("--lm_path", default="fast_agent/model_ckpts/flan_large_0402/checkpoint-300") 
        parser.add_argument("--simplification_str", default="easy")
        parser.add_argument("--beams", type=int, default=5)
        parser.add_argument("--max_episode_per_file", type=int, default=9999)
        parser.add_argument("--mode", default="fast_system")
        parser.add_argument("--set", default="test")
        parser.add_argument("--output_path", default="logs/test_fl-v1-300-bm=5")
        parser.add_argument("--compose_mode", default="v1_1")
        parser.add_argument("--model_parallelism_size", type=int, default=1)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--max_input_len", type=int, default=1024)
        parser.add_argument("--cut_off", action="store_true", default=True)
        parser.add_argument("--sbert", action="store_true", default=False)
        parser.add_argument("--no_stop", action="store_true", default=True) 

    args = parser.parse_args()
    params = vars(args)
    return params

#
#   Main
#

def init_logger(args, task_num, log_level=INFO):
    filenameOutPrefixSeed = get_file_name(args, task_num)
    logger = logging.getLogger()
    formatter = logging.Formatter("[%(asctime)s][%(levelname)s\t] %(message)s",
                                    datefmt='%Y-%m-%d %H:%M:%S')
    logger.setLevel(log_level)

    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logging_dir = args["output_path"]
    if logging_dir:
        os.makedirs(logging_dir, exist_ok=True)
        now = int(round(time.time() * 1000))
        timestr = time.strftime('%Y-%m-%d_%H-%M', time.localtime(now / 1000))
        filename = f"{filenameOutPrefixSeed}.log"
        fh = logging.FileHandler(filename)
        fh.setLevel(log_level)
        fh.setFormatter(formatter)
        if logger.hasHandlers():
            logger.handlers.clear()
        logger.addHandler(fh)
    return logger

def main():
    args = parse_args()
    print(args) 
    
    torch.manual_seed(args['seed'])
    torch.cuda.manual_seed(args['seed']) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    task_nums = args["task_nums"].split(",")
    for task_num in task_nums:
        logger = init_logger(args, task_num)
        logger.info(args)
        eval(args, int(task_num), logger)
        
if __name__ == "__main__":
    main()
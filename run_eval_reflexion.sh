for task in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29
do
    python eval_agent_reflexion.py \
        --task_nums $task \
        --set test_mini \
        --no_stop \
        --env_step_limit 100 \
        --simplification_str easy \
        --num_trials 10 \
        --prompt_file ReAct_baseline/prompt.jsonl \
        --output_path reflexion_logs/gpt-4 \
        --model_name gpt-4
done
for task in 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29
do
    python eval_agent_react.py \
        --task_nums $task \
        --set test_mini \
        --no_stop \
        --env_step_limit 100 \
        --simplification_str easy \
        --prompt_file ReAct_baseline/prompt.jsonl \
        --output_path ReAct_logs/gpt-4 \
        --model_name gpt-4
done
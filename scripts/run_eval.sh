# DEBUG_MODE="-m debugpy --listen 127.0.0.1:5679 --wait-for-client"

n_shards=1
swift_api_provider=vLLM
swift_model_id=/home/yifan/models/Llama-3.1-8B-Instruct
best_of_n=8
dataset=math-l5
prompt_template_dir=./swiftsage/prompt_templates

if [ $n_shards -eq 1 ]; then
    echo "n_shards = 1"
    python -m swiftsage.evaluate_benchmark \
        --swift_api_provider $swift_api_provider \
        --swift_model_id $swift_model_id \
        --dataset_name $dataset \
        --best_of_n $best_of_n \
        --prompt_template_dir $prompt_template_dir

elif [ $n_shards -gt 1 ]; then
    exp_id=$(cat /dev/urandom | head -n 10 | md5sum | cut -c 1-10)
    shards_dir="./output/math-l5/tmp_${exp_id}"
    echo "Using Data-parallelism. Output Path: ${shards_dir}"
    for ((shard_id = 0; shard_id < $n_shards; shard_id++)); do
        python -m swiftsage.evaluate_benchmark \
            --num_shards $n_shards \
            --shard_id $shard_id \
            --swift_api_provider $swift_api_provider \
            --swift_model_id $swift_model_id \
            --best_of_n $best_of_n \
            --dataset_name $dataset \
            --prompt_template_dir $prompt_template_dir \
            --output_path $shards_dir/ \
            --run_name $shard_id \
            &
    done 
    wait 
    # python src/merge_results.py $shards_dir/ $model_pretty_name
    # cp $shards_dir/${model_pretty_name}.json $output_dir/${model_pretty_name}.json
else
    echo "Invalid n_shards"
    exit
fi

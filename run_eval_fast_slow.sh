num_gpus=$1  
seed=42
split="test_mini"
gpt_version="gpt-3.5-turbo"
# gpt_version="gpt-4"


if [ $num_gpus -eq 1 ]; then
    task_nums=("13,19,1,14,27,10" "17,21,20,6,9,16" "0,4,2,8,7,11" "18,26,3,23,29,24" "12,25,22,5,15,28")
    L=1
elif [ $num_gpus -eq 4 ]; then
    task_nums=("0,12,20,16" "26,13,2,28" "22,17,3,10" "1,4,5,29" "18,14,11,15" "25,6,27,24" "19,8,9" "21,23,7")
    L=8
fi 

output_path="fast_slow_logs/${split}_all_0512_${gpt_version}/"
mkdir -p $output_path
echo "---> $output_path" 
 
cp eval_agent_fast_slow.py $output_path/
cp eval_utils.py $output_path/
cp slow_agent/demos.json $output_path/
cp data_utils/data_utils.py $output_path/

for ((i=0; i<L; i++)); do
    task_num=${task_nums[$i]}
    ((gpu=i%num_gpus)) # the number of gpus
    echo $task_num "on" $gpu    
    TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=$gpu python eval_agent_fast_slow.py \
        --task_nums $task_num \
        --set ${split} \
        --seed $seed \
        --debug_var -1 \
        --gpt_version $gpt_version \
        --output_path $output_path & # > /dev/null 2>&1 &
    sleep 10
done
 

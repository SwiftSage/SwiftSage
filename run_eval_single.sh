num_gpus=$1
ckpt=$2
split=$3
compose_mode=$4
beams=$5
seed=42


# ckpt=300
if [ $num_gpus -eq 1 ]; then
    task_nums=("12,25,22,5,15,28" "13,19,1,14,27,10" "17,21,20,6,9,16" "0,4,2,8,7,11" "18,26,3,23,29,24")
    L=5
    # task_nums=("12,25,22,5,15,28,13,19,1,14,27,10,17,21,20,6,9,16,0,4,2,8,7,11,18,26,3,23,29,24")
    # L=1 
elif [ $num_gpus -eq 4 ]; then
    # task_nums=("12,14" "13,6" "17,8" "4,23" "18,15" "25,27" "19,9" "21,7" "0,16" "26,28" "22,10" "1,29" "20,5" "2,3" "24" "11") # 16
    task_nums=("0,12,20,16" "26,13,2,28" "22,17,3,10" "1,4,5,29" "18,14,11,15" "25,6,27,24" "19,8,9" "21,23,7")
    L=8
fi 


# task_nums=("12,5" "13,14" "17,6" "4,8" "18,23" "25,15" "19,27" "21,9" "0,7" "26,16" "22,28" "1,10" "20,29" "2,11" "3,24")   # 15
# task_nums=("12,5,14" "13" "17,6" "4,8,15" "18,23" "25" "19,27,9" "21" "0,7" "26,16,10" "22,28" "1" "20,29" "2,11" "3,24") # 15-enhanced
# split=test_mini

if [ $compose_mode = "v1_1" ]; then 
    output_path=logs/${split}_fl-v1_1-$ckpt-bm=${beams}_sbert
    model_path=fast_agent/model_ckpts/flan_large_0402/checkpoint-$ckpt
    compose_mode=v1_1
elif [ $compose_mode = "v1" ]; then 
    output_path=logs/${split}_fl-v1-$ckpt-bm=${beams}_sbert
    model_path=fast_agent/model_ckpts/flan_large_0402/checkpoint-$ckpt
    compose_mode=v1_1
elif [ $compose_mode = "v3" ]; then 
    output_path=logs/${split}_fl-v3-$ckpt-bm=${beams}_nosbert
    model_path=fast_agent/model_ckpts/flan_large_0404/checkpoint-$ckpt
    compose_mode=v3
elif [ $compose_mode = "v4" ]; then 
    output_path=logs/${split}_fl-v4-$ckpt-bm=${beams}_sbert
    model_path=fast_agent/model_ckpts/flan_large_0405/checkpoint-$ckpt
    compose_mode=v4
elif [ $compose_mode = "v4.1" ]; then 
    # output_path=logs/${split}_fl-v4.1-$ckpt-bm=${beams}_sbert
    output_path=logs/${split}_fl-v4.1-$ckpt-bm=${beams}_nosbert
    model_path=fast_agent/model_ckpts/flan_large_0411/checkpoint-$ckpt
    compose_mode=v4
elif [ $compose_mode = "v5" ]; then 
    output_path=logs/${split}_fl-v5-$ckpt-bm=${beams}_sbert
    model_path=fast_agent/model_ckpts/flan_large_0413/checkpoint-$ckpt
    compose_mode=v5
fi

if [[ -e "$model_path/config.json" ]]; then
    echo "<--- $model_path"
else 
    echo "Error: $model_path"
fi   

mkdir -p $output_path
echo "---> $output_path"
#################### reproduce the best performance #################### 

########################################################################

#################### reproduce the best performance #################### 

########################################################################


# model_path=fast_agent/model_ckpts/flan_large_0404/checkpoint-$ckpt
# model_path=fast_agent/model_ckpts/t5_large/checkpoint-$ckpt
# model_path=fast_agent/model_ckpts/t5_large/checkpoint-300    
    


for ((i=0; i<L; i++)); do
    task_num=${task_nums[$i]}
    ((gpu=i%num_gpus)) # the number of gpus
    echo $task_num "on" $gpu    
    TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=$gpu python eval_agent_fast_only.py \
        --task_nums $task_num \
        --set ${split} \
        --cut_off --no_stop \
        --env_step_limit 100 \
        --lm_path $model_path \
        --simplification_str easy \
        --beams $beams \
        --seed $seed \
        --compose_mode $compose_mode \
        --output_path $output_path & # > /dev/null 2>&1 &
    sleep 5
done

# bash run_eval_single.sh 0 600
# bash run_eval_single.sh 1 700
# bash run_eval_single.sh 2 800
# bash run_eval_single.sh 3 900
# bash run_eval_single.sh 0 1000

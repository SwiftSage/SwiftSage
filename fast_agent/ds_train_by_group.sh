USE_TF=0
for tid in {1..10}
do
echo $tid
timeout 30m deepspeed --master_port 29512 \
		./ds_train.py \
		--cache_dir /net/nfs/path/to/cache/ \
        --model_name_or_path google/flan-t5-base \
        --output_dir model_ckpts/flan_base_${tid} \
        --do_train \
		--do_eval \
		--save_total_limit=3 \
        --train_file ../data_utils/data_dir/fast_system.${tid}.train.json \
		--validation_file ../data_utils/data_dir/fast_system.${tid}.val.mini.json \
		--predict_with_generate 0 \
        --learning_rate 1e-4 \
		--adam_eps 1e-06 \
        --overwrite_output_dir \
        --max_source_length 1024 \
        --max_target_length 16 \
        --per_device_train_batch_size 16 \
        --per_device_eval_batch_size 64 \
        --early_stopping_patience 3 \
        --metric_for_best_model eval_loss \
        --greater_is_better=False \
        --deepspeed zero_2_bf16.json \
        --gradient_accumulation_steps 3 \
        --num_train_epochs 30 \
        --logging_steps 1 \
        --load_best_model_at_end=True \
        --save_strategy=steps \
        --evaluation_strategy=steps \
        --save_steps 50 \
        --eval_steps 50 \
        --seed 42 \
        --report_to wandb \
        --run_name flan_base_${tid}
done
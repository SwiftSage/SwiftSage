DEBUG_MODE="-m debugpy --listen 127.0.0.1:5679 --wait-for-client"

python $DEBUG_MODE main.py \
    --eval_mode benchmark \
    --dataset_name MATH \
    --num_test_sample 4 \

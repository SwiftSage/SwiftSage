# DEBUG_MODE="-m debugpy --listen 127.0.0.1:5679 --wait-for-client"

# python $DEBUG_MODE -m swiftsage.evaluate_benchmark \
#     --dataset_name MATH \
#     --prompt_template_dir ./swiftsage/prompt_templates \
#     --num_test_sample 4 \

python $DEBUG_MODE -m swiftsage.evaluate_benchmark \
    --dataset_name gpqa \
    --prompt_template_dir ./swiftsage/prompt_templates \
    --num_test_sample 4 \

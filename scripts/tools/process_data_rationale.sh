BASE_PATH=${1}

export TF_CPP_MIN_LOG_LEVEL=3

# only prompt for MiniLLM train
PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/preprocess/process_data_dolly.py \
    --data-dir ${BASE_PATH}/data/rationale/ \
    --processed-data-dir ${BASE_PATH}/processed_data/rationale/prompt \
    --model-path ${BASE_PATH}/checkpoints/gpt2-xlarge \
    --data-process-workers 32 \
    --max-prompt-length 256 \
    --dev-num 1000 \
    --only-prompt \
    --model-type gpt2

# prompt and response for baselines
PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/preprocess/process_data_dolly.py \
    --data-dir ${BASE_PATH}/data/rationale/ \
    --processed-data-dir ${BASE_PATH}/processed_data/rationale/full \
    --model-path ${BASE_PATH}/checkpoints/gpt2-xlarge \
    --data-process-workers 32 \
    --max-prompt-length 256 \
    --dev-num 1000 \
    --model-type gpt2

base_path=${1-"/home/MiniLLM"}
port=2040


# Evaluate SFT
for seed in 10 20 30 40 50
do
    ckpt="sft/gpt2-base"
    bash ${base_path}/scripts/eval/eval_main_rationale.sh ${base_path} ${port} 1 ${ckpt} --seed $seed  --eval-batch-size 8
done

# # Evaluate KD
for seed in 10 20 30 40 50
do
    ckpt="kd/gpt2-base-xlarge-sft"
    bash ${base_path}/scripts/eval/eval_main_rationale.sh ${base_path} ${port} 1 ${ckpt} --seed $seed  --eval-batch-size 8
done

# # # Evaluate SeqKD
# for seed in 10 20 30 40 50
# do
#     ckpt="seqkd/gpt2-base-xlarge-sft"
#     bash ${base_path}/scripts/gpt2/eval/eval_main_${data}.sh ${base_path} ${port} 1 ${ckpt} --seed $seed  --eval-batch-size 8
# done

# # Evaluate MiniLLM
for seed in 10 20 30 40 50
do
    ckpt="minillm/base-init-xlarge-sft"
    bash ${base_path}/scripts/eval/eval_main_rationale.sh ${base_path} ${port} 1 ${ckpt} --seed $seed  --eval-batch-size 8
done
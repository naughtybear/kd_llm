# Comparison of Distillation Techniques for Language Model Compression
This project sets out to explore the performance differentials between white box distillation techniques and conventional black box distillation methods. It then aims to integrate white box distillation with a step-by-step paradigm.

## 1. Environment
```bash
bash install.sh
```

Our code is based in [this commit](https://github.com/huggingface/transformers/commit/85fde09c97213bf7e8625f83096bb2a9e183f987) of HuggingFace Transformers.

## 2. Data
### 2.1 Data Download
The finetune dataset "databricks-dolly-15k" can be download from the HugginFace datasaets [repository](https://huggingface.co/datasets/databricks/databricks-dolly-15k). The dowloaded file show change name to raw.jsonl and store it in 'data/dolly/'.

### 2.2 Data Processing
Tokenize the data and store them in binary files. Defualt valid size=1000, the rest is train data:
```bash
# Process Dolly Train / Validation Data
bash scripts/tools/process_data_dolly.sh /PATH/TO/Project
# Process Dolly+Rationale Train / Validation Data
bash scripts/tools/process_data_rational.sh /PATH/TO/Project
```

## 3. Model
### Based Pre-trained Models
To run fine-tuning or standard KD baselines, you need to download the model checkpoints from [Huggingface Model Hub] and put them in `checkpoints/`. For example, for gpt2-large, you can download the model from this [link](https://huggingface.co/gpt2-large/tree/main) and put them in `checkpoints/gpt2-large`.

## 4. Train
### 4.1 Baselines
Training the baseline model that compare to the white box KD and our method.
#### Fine-tune the teacher models
```bash
# train with dolly
bash scripts/sft/sft_xlarge.sh /PATH/TO/Project
# train with rationale
bash scripts/sft/sft_xlarge_rationale.sh /PATH/TO/Project
```

#### SFT(Supervise fine-tuning) Baselines
```bash
# train with dolly
bash scripts/gpt2/sft/sft_base.sh /PATH/TO/Project
# train with rationale
bash scripts/gpt2/sft/sft_base_rationale.sh /PATH/TO/Project
```

#### KD Baselines
```bash
# train with dolly
bash scripts/kd/kd_base.sh /PATH/TO/Project
# train with rationale
bash scripts/kd/kd_base_rationale.sh /PATH/TO/Project
```

### 4.2 MiniLLM
The white box KD method.
#### Initial Checkpoints
The final checkpoints are selected by the **validation loss**.
```bash
bash scripts/sft/sft_base.sh /PATH/TO/Project
```

#### Train
The final checkpoints are selected by the Rouge-L scores.
```bash
# train with dolly
bash scripts/minillm/train_base_xl.sh /PATH/TO/Project
# train with rationale
bash scripts/minillm/train_base_xl_rationale.sh /PATH/TO/Project
```

## 5. Run Evaluation
```bash
# evaluate the result without rationale
bash scripts/eval/run_eval.sh /PATH/TO/Project
# evaluate the result with rationale
bash scripts/eval/run_eval_rationale.sh /PATH/TO/Project
```

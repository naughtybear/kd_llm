import multiprocessing
import os
import time
import torch
import json
import sys
from numerize.numerize import numerize
import numpy as np
from data_utils.indexed_dataset import make_builder
from transformers import AutoTokenizer
from arguments import get_args


# 1. Implement an Encoder, which gives it a line of input data and it returns you the tokenized result.
class Encoder(object): 
    def __init__(self, args):
        self.args = args

    def initializer(self):
        Encoder.tokenizer = AutoTokenizer.from_pretrained(self.args.model_path)

    def encode(self, line):
        line = json.loads(line)
        # print(f"instruction:\n{line["instruction"]}")
        # print(f"context:\n{len(line["context"])}")
        # print("------")
        if "context" not in line or len(line["context"]) == 0:
            template = (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Response:\n"
            )
            prompt = template.format(instruction=line["instruction"])
        else:
            template = (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n###     :\n{input}\n\n### Response:\n"
            )
            prompt = template.format(instruction=line["instruction"], input=line["context"])
            
        response = line["response"]
        prompt_tokens = Encoder.tokenizer.encode(prompt, add_special_tokens=False)
        full_tokens = Encoder.tokenizer.encode(prompt + response, add_special_tokens=False) + [Encoder.tokenizer.eos_token_id]
        response_tokens = full_tokens[len(prompt_tokens):]
        
        if len(prompt_tokens) > self.args.max_prompt_length:
            return None, None, None, None, len(line)
        
        return line, prompt, prompt_tokens, response_tokens, len(line)


def main():
    print("OK")
    args = get_args()
        
    args.processed_data_dir = os.path.join(args.processed_data_dir, args.model_type)

    os.makedirs(args.processed_data_dir, exist_ok=True)
    
    with open(os.path.join(args.data_dir, "raw.jsonl")) as f:
        raw_data = f.readlines()

    if args.dev_num > 0:
        all_data = {
            "valid": raw_data[:args.dev_num],
            "train": raw_data[args.dev_num:]
        }
    else:
        all_data = {
            "train": raw_data
        }
    
    for split in all_data:
        
        # encoder use the tokenizer to encode data
        encoder = Encoder(args)

        # 2. Mapping all datas with Encoder, with the help of multiprocessing
        pool = multiprocessing.Pool(processes=args.data_process_workers, initializer=encoder.initializer)
        encoded_docs = pool.imap_unordered(encoder.encode, all_data[split], chunksize=50)
        proc_start = time.time()
        total_bytes_processed = 0
        
        bin_file = os.path.join(args.processed_data_dir, f"{split}_{0}.bin")
        idx_file = os.path.join(args.processed_data_dir, f"{split}_{0}.idx")

        binary_builder = make_builder(bin_file, impl="mmap", dtype=np.uint16)
        # binary_builder = make_builder(bin_file, impl="lazy", dtype=np.uint16)

        # put tokenized data into binary_builder
        inst_num = 0
        print("#"*10, split, "#"*10)
        
        prompt_lens = []
        response_lens = []
        
        json_file = open(os.path.join(args.processed_data_dir, f"{split}.jsonl"), "w")
        
        for lid, (line, prompt_str, prompt, response, bytes_processed) in enumerate(encoded_docs):
            # print(f"lid:{lid}")
            total_bytes_processed += bytes_processed
            if prompt is None:
                continue
            
            if args.only_prompt:
                if len(prompt) < args.max_length:
                    binary_builder.add_item(torch.IntTensor(prompt))
                else:
                    continue
            else:
                binary_builder.add_item(torch.IntTensor(prompt + [-1] + response))

            json_file.write(json.dumps({
                "instruction": line["instruction"],
                "prompt": prompt_str,
                "input": line["context"],
                "output": line["response"],
            }) + "\n")

            prompt_lens.append(len(prompt))
            response_lens.append(len(response))

            inst_num += 1
            if lid % 1000 == 0:
                current = time.time()
                elapsed = current - proc_start
                mbs = total_bytes_processed / elapsed / 1024 / 1024
                print(f"Processed {lid} documents. {inst_num} instances.",
                    f"({lid/elapsed} docs/s, {mbs} MB/s).",
                    file=sys.stderr)

        # finish compressing tokenized data into `bin_file`, and generate meta information into `idx_file`
        binary_builder.finalize(idx_file)

        # close multiproceessing mapping
        pool.close()
        json_file.close()
                
        print("Data num", len(prompt_lens))
        print("Prompt lengths.", "Mean:", np.mean(prompt_lens), "Max:", np.max(prompt_lens), "Min:", np.min(prompt_lens))
        print("Response", "Mean:", np.mean(response_lens), "Max:", np.max(response_lens), "Min:", np.min(response_lens))

def main_v2():
    print("start process dolly data")

    args = get_args()
        
    args.processed_data_dir = os.path.join(args.processed_data_dir, args.model_type)

    os.makedirs(args.processed_data_dir, exist_ok=True)
    
    with open(os.path.join(args.data_dir, "raw.jsonl")) as f:
        raw_data = f.readlines()

    # encoder use the tokenizer to encode data
    encoder = Encoder(args)

    # 2. Mapping all datas with Encoder, with the help of multiprocessing
    pool = multiprocessing.Pool(processes=args.data_process_workers, initializer=encoder.initializer)
    encoded_docs = pool.imap_unordered(encoder.encode, raw_data, chunksize=50)
    proc_start = time.time()
    total_bytes_processed = 0

    bin_file = os.path.join(args.processed_data_dir, f"{"valid"}_{0}.bin")
    idx_file = os.path.join(args.processed_data_dir, f"{"valid"}_{0}.idx")

    binary_builder = make_builder(bin_file, impl="mmap", dtype=np.uint16)
    # print(bin_file)
    # binary_builder = make_builder(bin_file, impl="lazy", dtype=np.uint16)
    # binary_builder.add_item(torch.IntTensor([21106, 318, 281, 12064, 326, 8477, 257, 4876, 13, 19430, 257, 2882, 326, 20431, 32543, 262, 2581, 13, 198, 198, 21017, 46486, 25, 198, 40, 2067, 257, 649, 1664, 3706, 13980, 6440, 3457, 13, 1867, 466, 356, 466, 30, 198, 198, 21017, 18261, 25, 198]))
    # print("success")

    # put tokenized data into binary_builder
    inst_num = 0
    print("#"*10, "valid", "#"*10)
    
    prompt_lens = []
    response_lens = []
    total_data = 0
    
    json_file = open(os.path.join(args.processed_data_dir, f"{"valid"}.jsonl"), "w")

    for lid, (line, prompt_str, prompt, response, bytes_processed) in enumerate(encoded_docs):
        # print(lid)
        if total_data == args.dev_num:
            print("-----------********----------")
            print(total_data)
            # finish compressing tokenized data into `bin_file`, and generate meta information into `idx_file`
            binary_builder.finalize(idx_file)

            json_file.close()

            print("Data num", len(prompt_lens))
            print("Prompt lengths.", "Mean:", np.mean(prompt_lens), "Max:", np.max(prompt_lens), "Min:", np.min(prompt_lens))
            print("Response", "Mean:", np.mean(response_lens), "Max:", np.max(response_lens), "Min:", np.min(response_lens))
                    
            proc_start = time.time()
            total_bytes_processed = 0
            
            bin_file = os.path.join(args.processed_data_dir, f"{"train"}_{0}.bin")
            idx_file = os.path.join(args.processed_data_dir, f"{"train"}_{0}.idx")

            binary_builder = make_builder(bin_file, impl="mmap", dtype=np.uint16)
            # binary_builder = make_builder(bin_file, impl="lazy", dtype=np.uint16)

            # put tokenized data into binary_builder
            inst_num = 0
            print("#"*10, "train", "#"*10)
            
            prompt_lens = []
            response_lens = []
            
            json_file = open(os.path.join(args.processed_data_dir, f"{"train"}.jsonl"), "w")
            
        total_bytes_processed += bytes_processed

        if prompt is None:
            continue
        
        if args.only_prompt:
            if len(prompt) < args.max_length:
                # print(prompt)
                total_data += 1
                binary_builder.add_item(torch.IntTensor(prompt))
            else:
                continue
        else:
            # print(prompt)
            # print(response)
            total_data += 1
            binary_builder.add_item(torch.IntTensor(prompt + [-1] + response))

        json_file.write(json.dumps({
            "instruction": line["instruction"],
            "prompt": prompt_str,
            "input": line["context"],
            "output": line["response"],
        }) + "\n")

        prompt_lens.append(len(prompt))
        response_lens.append(len(response))

        inst_num += 1
        if lid % 1000 == 0:
            current = time.time()
            elapsed = current - proc_start
            mbs = total_bytes_processed / elapsed / 1024 / 1024
            print(f"Processed {lid} documents. {inst_num} instances.",
                f"({lid/elapsed} docs/s, {mbs} MB/s).",
                file=sys.stderr)
                
    # finish compressing tokenized data into `bin_file`, and generate meta information into `idx_file`
    binary_builder.finalize(idx_file)

    # close multiproceessing mapping
    pool.close()
    json_file.close()
            
    print("Data num", len(prompt_lens))
    print("Prompt lengths.", "Mean:", np.mean(prompt_lens), "Max:", np.max(prompt_lens), "Min:", np.min(prompt_lens))
    print("Response", "Mean:", np.mean(response_lens), "Max:", np.max(response_lens), "Min:", np.min(response_lens))

if __name__ == '__main__':
    main_v2()
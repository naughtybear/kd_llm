import numpy as np
import pandas as pd
import torch

# Generate Rationales
def generate_rationales(data_path, rationale_prompt, model_name, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df = pd.read_csv(data_path)
    df = df.drop('context', axis=1)

    def add_punctuation(text):
        if text[-1] not in ['.', '?']:
            text += '.'
        return text

    # Apply the function to the 'text' column
    df['instruction'] = df['instruction'].apply(add_punctuation)
    df['response'] = df['response'].apply(add_punctuation)

    # Create reformated column
    start = rationale_prompt
    df['llm_input'] = df.apply(lambda row: start  + "Instruction: " + str(row['instruction']) + " Response: " + str(row['response']), axis=1)
    #df.to_csv("test.csv", index=False)
    
    # Import model
    if model_name == 'gpt2':
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
        model = GPT2LMHeadModel.from_pretrained('gpt2-large', pad_token_id = tokenizer.eos_token_id)
    elif model_name =='gpt_neo':
        from transformers import GPTNeoForCausalLM, GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
        model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B", pad_token_id = tokenizer.eos_token_id)

    model.to(device) # Send to GPU
    llm_outputs = []

    print(f"Starting Generation")
    for index, row in df.iterrows():
        llm_input = row['llm_input']
        input_ids = tokenizer.encode(llm_input, return_tensors='pt').to(device)
        input_length = len(input_ids[0])
        
        print(f"\t({index}/{len(df)-1}) Tok_len: {input_length}")
        if input_length > 870:
            print(f"\t\tInput exceeds GPT2 max length of 1024")
            llm_outputs.append('NaN')
            #input('stop:')
        else:
            max_output_length = input_length + 150 #+ int(input_length*0.8)
            print(f"\t\tMax_len_out: {max_output_length}")
            if max_output_length > 1023:
                max_output_length = 1023
            
            try:
                if model_name == 'gpt2':
                    output = model.generate(input_ids,
                                            max_length = max_output_length,
                                            num_beams = 5,
                                            do_sample=True,
                                            temperature=0.1,
                                            no_repeat_ngram_size = 2,
                                            early_stopping = True,
                                            )
                elif model_name == 'gpt_neo':
                    attention_mask = torch.ones_like(input_ids)
                    output = model.generate(input_ids,
                                            max_length = max_output_length,
                                            num_beams = 5,
                                            do_sample=True,
                                            temperature=0.1,
                                            no_repeat_ngram_size = 2,
                                            early_stopping = True,
                                            attention_mask=attention_mask
                                            )
                
                output_decoded = tokenizer.decode(output[0])
                llm_outputs.append(output_decoded)
            except Exception as e:
                print(f"\t\tError occured: {e}")
                output_decoded = 'NaN'
                llm_outputs.append(output_decoded)
                
    df['llm_output'] = llm_outputs
    df.to_csv(output_path, index=False)


data_path = "./csv/databricks_nocontext.csv"
rationale_prompt = "Rationalize the relationship between the following: "
model_name = 'gpt_neo'
output_path = 'gptneo_output.csv'

generate_rationales(data_path, 
                    rationale_prompt, 
                    model_name, 
                    output_path)
import pandas as pd
import re

# Remove incomplete sentences
def remove_incomplete_sentence(text):
    sentence_pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s'
    sentences = re.split(sentence_pattern, text)
    
    if sentences:
        last_sentence = sentences[-1]
        if last_sentence and not last_sentence.endswith('.'):
            sentences = sentences[:-1]

    return ' '.join(sentences)

# Convert LLM rationales into JSON format
def csv_to_json(csv_path, json_path):
    df = pd.read_csv(csv_path)

    # Drop rows which had NaN appended to llm_output
    print(f"Before drop: {len(df)}")
    df.dropna(subset=['llm_output'], inplace=True)
    print(f"After drop: {len(df)}")

    # Remove input from output column
    df['len_in'] = df['llm_input'].apply(lambda x: len(x))
    df['llm_output'] = df.apply(lambda row: row['llm_output'][row['len_in']:], axis=1)
    df.drop('len_in', axis=1, inplace=True)

    # Strip /n + eos token characters from output column
    df['llm_output'] = df['llm_output'].str.lstrip() # from beginning
    df['llm_output'] = df['llm_output'].str.replace('\n', ' ')
    rem_chars = ['<|endoftext|>', '\n'] # from end
    for char in rem_chars:
        df['llm_output'] = df['llm_output'].str.rstrip(char)

    df['llm_output'] = df['llm_output'].apply(remove_incomplete_sentence)

    # Remove empty rows and reset index
    df = df[df['llm_output'] != '']
    df.reset_index(drop=True, inplace=True) # Reset index

    columns = ['instruction', 'llm_output', 'response', 'category']
    df_json = df[columns]
    new_column_names = {'llm_output': 'context'}
    df_json = df_json.rename(columns=new_column_names)

    # Convert JSON format
    jsonl_data = df_json.to_json(orient='records', lines=True)
    with open(json_path, 'w') as f:
        f.write(jsonl_data)

csv_to_json('gpt2_output.csv', 'gpt2_output.jsonl')
import pandas as pd
from vllm import LLM, SamplingParams
import glob
import re
import pathlib
import os
import argparse

def clean_citations(text):
    # Remove text between <ref and </ref>
    return re.sub(r'<ref.*?</ref>', '', text, flags=re.DOTALL)

def extract_content(text, start_tag, end_tag):
    pattern = f"{start_tag}(.*?){end_tag}"
    match = re.search(pattern, text, re.DOTALL)  # re.DOTALL allows matching across multiple lines
    return match.group(1).strip() if match else None

def extract_generated_components(text):
    # Extract each component using regex
    query_analysis = re.search(r'### Query analysis ###\n(.*?)\n\n###', text, re.DOTALL)
    query_analysis = query_analysis.group(1).strip() if query_analysis else ""
    
    query_adherence = re.search(r'### Query adherence ###\n(.*?)\n\n###', text, re.DOTALL)
    query_adherence = query_adherence.group(1).strip() if query_adherence else ""
    
    answer_analysis = re.search(r'### Answer analysis ###\n(.*?)\n\n###', text, re.DOTALL)
    answer_analysis = answer_analysis.group(1).strip() if answer_analysis else ""
    
    language_quality = re.search(r'### Language quality ###\n(.*?)\n\n', text, re.DOTALL)
    language_quality = language_quality.group(1).strip() if language_quality else ""
    
    reasoning_quality = re.search(r'### Reasoning quality ###\n(.*?)(?:\n\n|$)', text, re.DOTALL)
    reasoning_quality = reasoning_quality.group(1).strip() if reasoning_quality else ""
    
    return {
        'query_analysis': query_analysis,
        'query_adherence': query_adherence,
        'answer_analysis': answer_analysis,
        'language_quality': language_quality,
        'reasoning_quality': reasoning_quality
    }


parser = argparse.ArgumentParser(description='Process parquet files from a directory.')
parser.add_argument('base_path', type=str, help='Base path to search for parquet files')
    
# Parse arguments
args = parser.parse_args()

print(args.base_path)

list_files = glob.glob(args.base_path + "/*parquet")

print(list_files)

# Load model and set sampling parameters
llm = LLM("llama-rag-eval/llama-rag-eval", max_model_len=8128)
sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=3000, presence_penalty=1.2, stop=["#END#"])

for file in list_files:

    final_file = file.replace("/lustre/fswork/projects/rech/fmr/uft12cr/corpus_rag/parquets/", "/lustre/fsn1/projects/rech/fmr/uft12cr/corpus_rag_evaluated/")

    if os.path.exists(final_file):

        print(file + " already created using the next one")
    
    else:

        directory = os.path.dirname(final_file)
        
        pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
        
        result = pd.read_parquet(file)

        print(result)
        
        texts = result["text"].tolist()
        list_ids = result["chunk_id"].tolist()

        list_texts = []
        
        for text in texts:
            query = extract_content(text, "<\|query_start\|>", "<\|query_end\|>")
            answer = extract_content(text, "<\|answer_start\|>", "<\|answer_end\|>")

            answer = clean_citations(answer)

            combined_text = f'### Query ###\n{query}\n\n### Answer ###\n{answer}\n\n### Query analysis ###\n'

            list_texts.append(combined_text)
        
        outputs = llm.generate(list_texts, sampling_params)
        generated_texts = [output.outputs[0].text for output in outputs]

        components = [extract_generated_components(text) for text in generated_texts]

            # Create the DataFrame with all columns
        df = pd.DataFrame({
            'chunk_id': list_ids,
            'original_text': list_texts,
            'analysis': generated_texts,
            'query_analysis': [comp['query_analysis'] for comp in components],
            'query_adherence': [comp['query_adherence'] for comp in components],
            'answer_analysis': [comp['answer_analysis'] for comp in components],
            'language_quality': [comp['language_quality'] for comp in components],
            'reasoning_quality': [comp['reasoning_quality'] for comp in components]
        })

        df.to_parquet(final_file)
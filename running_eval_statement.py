import pandas as pd
from vllm import LLM, SamplingParams
import glob
import re
import pathlib
import os
import argparse

def clean_dataset(df):
    """
    Clean the dataset by removing quotes from citations and cleaning up statement prefixes.
    
    Args:
        df (pandas.DataFrame): Input DataFrame with 'citation' and 'statement' columns
        
    Returns:
        pandas.DataFrame: Cleaned DataFrame
    """
    # Create a copy to avoid modifying the original DataFrame
    result = df.copy()
    
    # Remove quotes from citations
    result['citation'] = result['citation'].str.replace('"', '', regex=False)
    
    # Clean up statement prefixes
    # Using a single regex pattern to match all prefixes
    prefixes_pattern = r'^(\. |\)\. |\* |\*\*\) )'
    result['statement'] = result['statement'].str.replace(prefixes_pattern, '', regex=True)
    
    return result

def extract_content(text, start_tag, end_tag):
    pattern = f"{start_tag}(.*?){end_tag}"
    match = re.search(pattern, text, re.DOTALL)  # re.DOTALL allows matching across multiple lines
    return match.group(1).strip() if match else None

def get_grounding_context(text, ref_start_pos):
    """
    Extract grounding context before a reference based on specific rules:
    1. Stop at newline (paragraph boundary)
    2. Stop at previous reference
    3. Limit to roughly 3 sentences
    
    Args:
        text (str): Full text
        ref_start_pos (int): Starting position of the reference
        
    Returns:
        str: Extracted grounding context
    """
    # Look backwards from the reference position
    current_pos = ref_start_pos
    
    # Find first newline before reference
    newline_boundary = text.rfind('\n', 0, current_pos)
    
    # Find previous reference end
    prev_ref_boundary = text.rfind('</ref>', 0, current_pos)
    
    # If we found a previous reference, we need to look after its end
    if prev_ref_boundary != -1:
        prev_ref_boundary += len('</ref>')
    
    # Take the later boundary between newline and previous reference
    effective_boundary = max(newline_boundary, prev_ref_boundary)
    
    # If no boundary found, start from beginning
    if effective_boundary == -1:
        effective_boundary = 0
    
    # Extract the text between boundary and reference
    context = text[effective_boundary:ref_start_pos].strip()
    
    # Limit to roughly 3 sentences if needed
    sentences = re.split(r'(?<=[.!?])\s+', context)
    if len(sentences) > 3:
        context = ' '.join(sentences[-3:])
    
    return context.strip()

def extract_references(text, generation_id):
    """
    Extract references with comprehensive grounding context.
    
    Args:
        text (str): Input text containing references
        
    Returns:
        tuple: (list of reference dictionaries, pandas DataFrame)
    """
    references = []
    
    # Pattern to match reference tags and their content
    ref_pattern = r'<ref\s+name="([^"]+)">([^<]+)<\/ref>'
    
    # Find all references in the text
    for match in re.finditer(ref_pattern, text):
        ref_id = match.group(1)      # The reference ID
        citation = match.group(2)     # The actual citation text
        
        # Get the grounding context using the new rules
        grounding_context = get_grounding_context(text, match.start())
        
        reference_obj = {
                'generation_id': generation_id,
                'statement': grounding_context,
                'statement_size': len(grounding_context.split()),
                'reference_id': ref_id,
                'citation': citation,
                'citation_size': len(citation.split()),
        }
        
        references.append(reference_obj)
    
    return references

def extract_generated_components(text):
    # Extract each component using regex
    analysis = re.search(r'### Analysis ###\n(.*?)\n\n###', text, re.DOTALL)
    analysis = analysis.group(1).strip() if analysis else ""
    
    judgement = re.search(r'### Judgement ###\n(.*?)(?:\n\n|$)', text, re.DOTALL)
    judgement = judgement.group(1).strip() if judgement else ""
    
    return {
        'analysis': analysis,
        'judgement': judgement
    }

parser = argparse.ArgumentParser(description='Process parquet files from a directory.')
parser.add_argument('base_path', type=str, help='Base path to search for parquet files')
    
# Parse arguments
args = parser.parse_args()

print(args.base_path)

list_files = glob.glob(args.base_path + "/**/*parquet", recursive=True)

print(list_files)

# Load model and set sampling parameters
llm = LLM("llama-rag-eval/llama-rag-eval", max_model_len=8128)
sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=3000, presence_penalty=1.2, stop=["#END#"])

for file in list_files:

    final_file = file.replace("/lustre/fswork/projects/rech/fmr/uft12cr/corpus_rag/parquets/", "/lustre/fsn1/projects/rech/fmr/uft12cr/citation_rag_evaluated/")

    if os.path.exists(final_file):
        
        print(final_file + " already is there.")

    else:

        directory = os.path.dirname(final_file)
        
        pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
    
        result = pd.read_parquet(file)

        structured_reference_set = []
        
        for ind, row in result.iterrows():
            answer = extract_content(row["text"], "<\|answer_start\|>", "<\|answer_end\|>")

            references = extract_references(answer, row["chunk_id"])
            structured_reference_set.extend(references)

        structured_reference_set = pd.DataFrame(structured_reference_set)

        structured_reference_set = clean_dataset(structured_reference_set)

        list_texts = []

        for ind, row in structured_reference_set.iterrows():
            list_texts.append("### Statement ###\n" + row["statement"] + "\n\n### Citation ###\n" + row["citation"] + "\n\n### Analysis ###\n")

        outputs = llm.generate(list_texts, sampling_params)
        generated_texts = [output.outputs[0].text for output in outputs]

        components = [extract_generated_components(text) for text in generated_texts]

        structured_reference_set['analysis'] = [comp['analysis'] for comp in components]
        structured_reference_set['judgement'] = [comp['judgement'] for comp in components]
        
        structured_reference_set.to_parquet(final_file)
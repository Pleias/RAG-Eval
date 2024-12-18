import json, re
import pprint
import pandas as pd


import re

def extract_references(text, generation_id, model):
    """
    Extract references, their IDs, and grounded statements from text.
    
    Args:
        text (str): Input text containing references in Wikipedia-style format
        
    Returns:
        list: List of dictionaries containing reference information
    """
    # Split text into sentences (basic split on periods followed by space)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    references = []
    
    # Pattern to match reference tags and their content
    ref_pattern = r'<ref\s+name="([^"]+)">([^<]+)<\/ref>'
    
    # Iterate through sentences to find references and their context
    for i, sentence in enumerate(sentences):
        # Find all references in the current sentence
        refs = re.finditer(ref_pattern, sentence)
        
        for ref in refs:
            ref_id = ref.group(1)      # The reference ID
            citation = ref.group(2)     # The actual citation text
            
            # Get the grounding statement (sentence before the reference)
            # Remove the reference tag from the current sentence to get clean text
            clean_sentence = re.sub(ref_pattern, '', sentence).strip()
            
            reference_obj = {
                'generation_id': generation_id,
                'model': model,
                'statement': clean_sentence,
                'statement_size': len(clean_sentence.split()),
                'reference_id': ref_id,
                'citation': citation,
                'citation_size': len(citation.split()),
            }
            
            references.append(reference_obj)
    
    return references


reference_set = open("llm_evaluations.json", 'r')

reference_set = json.load(reference_set)

structured_reference_set = []

#Extraction in a structured way
for instruction in reference_set[0:]:
    generation_id, model, source, analysis = instruction["generation_id"], instruction["model"], instruction["text"], instruction["generated_response"]

    references = extract_references(analysis, generation_id, model)
    structured_reference_set.extend(references)

structured_reference_set = pd.DataFrame(structured_reference_set)
structured_reference_set.to_parquet("eval_statement_set.parquet")
    
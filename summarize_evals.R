library(tidyverse)
library(tidytext)

eval = arrow::read_parquet("evaluation_consolidated.parquet")
eval = eval %>% group_by(model) %>% mutate(generation_id = 1:n()) %>% ungroup() %>% mutate(generation_id = paste0(model, "_", generation_id))

#We extract the texts of the sources.
sources_pleias = eval %>%
  filter(grepl("Pleias", model)) %>%
  mutate(source_text = str_extract_all(text, regex("<\\|source_start\\|>.+?<\\|source_end\\|>", dotall=TRUE))) %>%
  select(generation_id, model, source_text) %>%
  unnest() %>%
  mutate(source_id = str_extract(source_text, "<\\|source_id_start\\|>.+?<\\|source_id_end\\|>")) %>%
  mutate(source_id = gsub("<\\|.+?\\|>", "", source_id)) %>%
  mutate(source_text = gsub("<\\|source_id_start\\|>.+?<\\|source_id_end\\|>", "", source_text)) %>%
  mutate(source_text = gsub("<\\|.+?\\|>", "", source_text))

sources_other = eval %>%
  filter(!grepl("Pleias", model)) %>%
  mutate(source_text = gsub(regex("^.+?You can take information from the following texts:\n", dotall=TRUE), "", text)) %>%
  mutate(source_text = gsub(regex("\n\nFinally, your answer should be written.+?$", dotall=TRUE), "", source_text)) %>%
  mutate(source_text = strsplit(source_text, "\n\\*\\*")) %>%
  select(generation_id, model, source_text) %>%
  unnest() %>%
  mutate(source_id = str_extract(source_text, "^.+?\\*\\*")) %>%
  mutate(source_id = gsub("\\*\\*", "", source_id)) %>%
  mutate(source_text = gsub("^.+?\\*\\*\n", "", source_text))

sources_texts = bind_rows(sources_pleias, sources_other)

#We extract the texts from the references.
reference_texts = eval %>%
  mutate(reference = str_extract_all(generated_response, '<ref name.+?</ref>')) %>%
  select(generation_id, model, reference) %>%
  unnest() %>%
  mutate(citation = str_extract(reference, '">.+?</ref>')) %>%
  mutate(citation = gsub('">|</ref>|"', '', citation)) %>%
  mutate(reference_id = str_extract(reference, '<ref name=".+?">')) %>%
  mutate(reference_id = gsub('">|<ref name="', '', reference_id)) %>%
  select(-reference) %>%
  group_by(generation_id) %>%
  mutate(citation_id = paste0(generation_id, "_citation_", 1:n())) %>%
  ungroup()

#First test: correct reference_ids:
correct_citations = reference_texts %>%
  select(generation_id, model, reference_id, citation_id) %>%
  inner_join(sources_texts %>% select(generation_id, source_id)) %>%
  filter(reference_id == source_id) %>%
  distinct(citation_id)

total_ids = reference_texts %>% count(model, generation_id, name = "cited_id")

hallucination_ranking = reference_texts %>%
  anti_join(correct_citations) %>%
  group_by(model, generation_id) %>%
  summarise(hallucinated_id = n())

correct_identifier = total_ids %>%
  left_join(hallucination_ranking) %>%
  mutate(hallucinated_id = ifelse(is.na(hallucinated_id), 0, hallucinated_id)) %>%
  group_by(model) %>%
  summarise(cited_id = sum(cited_id), hallucinated_id = sum(hallucinated_id)) %>%
  mutate(correct_id = 1-(hallucinated_id/cited_id)) %>%
  arrange(-correct_id)

#Second test: working quotes (regardless of whether they are hallucinated)
bad_quotes = reference_texts %>%
  mutate(citation_length = str_count(citation, "\\S+")) %>%
  filter(citation_length <= 2) %>%
  distinct(citation_id)

bad_quotes_ranking = reference_texts %>%
  inner_join(bad_quotes) %>%
  group_by(model, generation_id) %>%
  summarise(bad_quote = n())

valid_quote = total_ids %>%
  filter(model != "smollm_1.7b_instruct") %>%
  left_join(bad_quotes_ranking) %>%
  mutate(bad_quote = ifelse(is.na(bad_quote), 0, bad_quote)) %>%
  group_by(model) %>%
  summarise(cited_id = sum(cited_id), bad_quote = sum(bad_quote)) %>%
  mutate(valid_quote_ratio = 1-(bad_quote/cited_id)) %>%
  arrange(-valid_quote_ratio)

#Third text: share of repeated quotes.
duplicated_citation = reference_texts %>%
  select(generation_id, citation_id, citation) %>%
  inner_join(reference_texts %>% select(generation_id, match_citation_id = citation_id, citation)) %>%
  filter(citation_id != match_citation_id) %>%
  distinct(citation_id)

duplicated_citation_ranking = reference_texts %>%
  anti_join(bad_quotes) %>%
  inner_join(duplicated_citation) %>%
  group_by(model, generation_id) %>%
  summarise(duplicated_quote = n()) %>%
  mutate(duplicated_quote = duplicated_quote/2)

duplicated_citation = total_ids %>%
  filter(model != "smollm_1.7b_instruct") %>%
  left_join(duplicated_citation_ranking) %>%
  mutate(duplicated_quote = ifelse(is.na(duplicated_quote), 0, duplicated_quote)) %>%
  group_by(model) %>%
  summarise(cited_id = sum(cited_id), duplicated_quote = sum(duplicated_quote)) %>%
  mutate(unduplicated_quote_ratio = 1-(duplicated_quote/cited_id)) %>%
  arrange(-unduplicated_quote_ratio)

#Heart of the matter: Hallucinated citations.

get_align <- function(source_a, source_b){
  tryCatch({
    alignement <- textreuse::align_local(source_a, source_b)
    score = alignement$score
    a_edits = alignement$a_edits
    b_edits = alignement$b_edits
    alignement = tibble(score = score, a_edits = a_edits, b_edits = b_edits)
    return(alignement)
  }, error = function(e) {
    # If an error occurs, execute this code block
    # Returning a tibble with 0 and NA to keep the structure but indicate failure
    return(tibble(score = 0, a_edits = NA, b_edits = NA))
  })
}

#To save a bit on computation we take the unique citations.
unique_reference_texts = reference_texts %>%
  distinct(generation_id, citation, .keep_all = TRUE)

match_citation = unique_reference_texts %>%
  inner_join(sources_texts %>% select(reference_id = source_id, generation_id, source_text))

alignment_citation = match_citation %>%
  rowwise() %>%
  do(result = get_align(.$citation, .$source_text)) %>%
  unnest()

#Check why references are duplicated?
match_citation = match_citation %>%
  mutate(score = alignment_citation$score) %>%
  group_by(citation_id) %>%
  filter(score == max(score)) %>%
  ungroup() %>%
  mutate(size_citation = str_count(citation, "\\S+")) %>%
  mutate(match_score = ifelse(score > 0, (score/2)/size_citation, 0))

non_hallucinated_citation = match_citation %>%
  group_by(model) %>%
  summarise(non_hallucinated_citation = mean(match_score)) %>%
  arrange(-non_hallucinated_citation)

rag_benchmark = correct_identifier %>% select(-cited_id) %>%
  inner_join(valid_quote) %>% select(-cited_id) %>%
  inner_join(duplicated_citation) %>% select(-cited_id) %>%
  inner_join(non_hallucinated_citation) %>%
  select(-hallucinated_id, -bad_quote, -duplicated_quote) %>%
  mutate(rag_rank = (correct_id + valid_quote_ratio + unduplicated_quote_ratio + non_hallucinated_citation)/4) %>%
  arrange(-rag_rank)


reference_texts %>%
  mutate(citation_length = str_count(citation, "\\S+")) %>%
  group_by(model, generation_id) %>%
  summarise(citations = n(), citation_length = mean(citation_length)) %>%
  summarise(citations = mean(citations, na.rm = TRUE), citation_length = mean(citation_length, na.rm = TRUE)) %>%
  arrange(-citations) %>%
  View()

#Working quotes (not just a reference or empty)
#Repeated quotes.
#Hallucinated ids.
#Hallucinated texts.

#Sentence/Statement extraction.

reference_statement = eval %>%
  select(generation_id, model, generated_response) %>%
  mutate(generated_response = gsub("<ref ", ". A<ref ", generated_response)) %>%
  mutate(generated_response = gsub("</ref>", "</ref>. AZRT", generated_response)) %>%
  unnest_sentences(sentence, generated_response, to_lower = FALSE) %>%
  mutate(sentence = gsub("A<ref", "<ref", sentence)) %>%
  mutate(sentence = gsub("^AZRT", "", sentence)) %>%
  mutate(type_sentence = ifelse(grepl("^<ref", sentence), "reference", NA)) %>%
  mutate(type_sentence = ifelse(grepl("</ref>", lag(sentence)), "text", type_sentence)) %>%
  group_by(generation_id) %>%
  fill(type_sentence) %>%
  mutate(type_sentence = ifelse(is.na(type_sentence), "text", type_sentence)) %>%
  filter(sentence != "</ref>.") %>%
  mutate(sentence_id = 1:n())

library(dplyr)
library(stringr)

merge_consecutive_references <- function(df) {
  # Create a group identifier for consecutive references
  df <- df %>%
    group_by(generation_id) %>%
    mutate(
      # Create a new group whenever the type changes
      ref_group = cumsum(
        type_sentence == "reference" & 
          (lag(type_sentence, default = "none") != "reference")
      ),
      # Only group references, other rows get NA
      ref_group = if_else(type_sentence == "reference", ref_group, NA_real_)
    )
  
  # Process references
  ref_merged <- df %>%
    filter(type_sentence == "reference") %>%
    group_by(generation_id, ref_group) %>%
    summarise(
      sentence = paste(sentence, collapse = " "),
      type_sentence = "reference",
      model = first(model),
      sentence_id = first(sentence_id),
      .groups = "drop"
    )
  
  # Combine with non-reference rows
  result <- df %>%
    filter(type_sentence != "reference") %>%
    select(-ref_group) %>%
    bind_rows(ref_merged) %>%
    arrange(generation_id, sentence_id)
  
  return(result)
}

merged_reference = merge_continuous_references(reference_statement %>% ungroup(generation_id))





#We get the references and their associated texts.
reference_target = reference_statement %>%
  filter(type_sentence == "reference")

statement_target = reference_statement %>%
  filter(lead(type_sentence) == "reference") %>%
  filter(type_sentence == "text")

reference_statement_extract = reference_target %>%
  bind_rows(statement_target) %>%
  arrange(generation_id, sentence_id) %>%
  mutate(sentence = gsub(regex('<ref .+?>', dotall=TRUE), '', sentence)) %>%
  mutate(sentence = gsub('</ref>|\\"', "", sentence))

reference_statement_extract %>%
  mutate(reference_text = str_extract())

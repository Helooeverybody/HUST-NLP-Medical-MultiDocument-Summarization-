import torch
import spacy
import numpy as np
from collections import defaultdict
from rouge_score import rouge_scorer
import random
from transformers import AutoTokenizer 
nlp = spacy.load("en_core_web_sm")
  # NER model from spacy to extract entities later
def truncate(all_docs, max_length_input, mask_ratio, non_mask_ratio):
    # Truncate the documents to desired
    truncated_docs = []
    for doc in all_docs:
        cur_idx = 0
        all_sents = []
        for s in doc:
            cur_idx += len(s.split())
            # if add the new sentence exceeds the expected length limit, don't add it
            if cur_idx >= (
                (max_length_input * (1 + mask_ratio * (1 - non_mask_ratio)))
                // len(all_docs)
            ):
                break
            all_sents.append(s)
        truncated_docs.append(all_sents)
    return truncated_docs
def get_entities(all_docs):
    """
    Extract named entities and compute entity importance scores.
    Args:
        all_docs : List of docs, where each doc is a list of sents.

    Returns:
        desceding_sorted_dict: entity: n(e) for n(e) is number of docs entities appear 
    """
    entity_pyramid = defaultdict(int)

    for doc in all_docs:
        doc_entities = set()  
        for sent in doc:
            sent = nlp(sent)
            for ent in sent.ents:
                doc_entities.add(ent.text.lower())  

        for ent in doc_entities:
            entity_pyramid[ent] += 1

    sorted_entities = sorted(entity_pyramid.items(), key=lambda x: x[1], reverse=True)
    return dict(sorted_entities)  
def compute_rouge_scores(all_docs,scorer):
    """
    Compute ROUGE scores for each sentence .

    Args:
        all_docs (list of list of str): List of documents (each document is a list of sentences).

    Returns:
        dict: Mapping (doc_idx, sent_idx) → ROUGE score.
    """
    rouge_scores_dict = {}
    
    for doc_idx, doc in enumerate(all_docs):
        for sent_idx, sent in enumerate(doc):
            # Compute ROUGE score against full document (excluding itself)
            ref_text = " ".join([s for i, s in enumerate(doc) if i != sent_idx])
            rouge_scores = scorer.score(sent, ref_text)

            # Compute average ROUGE score
            avg_rouge_score = (rouge_scores["rouge1"].fmeasure +rouge_scores["rouge2"].fmeasure +rouge_scores["rougeL"].fmeasure) / 3

            # Store in dictionary
            rouge_scores_dict[sent] = avg_rouge_score

    return rouge_scores_dict
def select_salient_sentences(truncated_docs,rouge_score_dict, entity_pyramid, num_sents_to_mask):
    """
    Select sentences to mask based on entity importance and ROUGE scores.

    Args:
        all_docs (list of list of str): List of documents (each document is a list of sentences).
        entity_pyramid (dict): Entities sorted by document frequency.
        rouge_score_dict (dict- already sorted in descending): sent- rouge score(sent) 
        num_sents_to_mask (int): Number of sentences to mask.

    Returns:
        list of (doc_idx, sent_idx): Indices of selected sentences to mask.
    """
    selected_sents = set()
    entity_to_sent = {}
    
    # Iterate over entities (sorted by importance)
    for entity, appearance in entity_pyramid.items():
        if appearance==1: #ignore all entity that appear in only 1 document
            break
        best_sentence = None
        best_score = -1
        best_doc_idx, best_sent_idx = None, None

        # Find the best sentence containing this entity
        for doc_idx, doc in enumerate(truncated_docs):
            for sent_idx, sent in enumerate(doc):
                if entity in sent.lower():  
                    score=rouge_score_dict[sent]
                    # Keep track of the highest pyramid score sentence per entity
                    if score > best_score:
                        #best_sentence = sent
                        #best_score = score
                        best_doc_idx, best_sent_idx = doc_idx, sent_idx

        if best_sentence:
            #entity_to_sent[entity] = (best_doc_idx, best_sent_idx, best_score)
            selected_sents.add((best_doc_idx,best_sent_idx))
        # Stop when select enough sentences
        if len(selected_sents) >= num_sents_to_mask:
            break
    
    if len(selected_sents)<num_sents_to_mask: #if not enough sentences then keep appending highest pyramid score sentences until enough
        map_sent_to_idx={} #map sentence text to its (doc_idx,sen_idx) in truncated doc
        for doc_idx,doc in enumerate(truncated_docs):
            for sent_idx,sent in enumerate(doc):
                map_sent_to_idx[sent]=(doc_idx,sent_idx) 
        for sent in rouge_score_dict.keys():
            if sent not in map_sent_to_idx:
                continue
            selected_sents.add(map_sent_to_idx[sent])
            if len(selected_sents)==num_sents_to_mask:
                break
    return list(selected_sents)

def get_src_tgt_and_mask(truncated_docs, selected_sents,tokenizer,max_len_input,max_len_output,non_mask_ratio):
    """
    Get source and tgt

    Args:
        truncated_docs (list of list of str): list of documents with sentences.
        selected_sents (list of (doc_idx, sent_idx)): Indices of sentences to mask.

    Returns:
        Src: the cluster with masked salient sentences
        Target: The masked salient sentences
    """
    non_mask_sents = random.sample(
            list(selected_sents), int(len(selected_sents) * non_mask_ratio)
        )
    masked_docs = [doc.copy() for doc in truncated_docs] 
    tgt=[]
    for doc_idx, sent_idx in selected_sents:
        tgt.append(truncated_docs[doc_idx][sent_idx])
        if (doc_idx,sent_idx) in non_mask_sents: 
            continue
        masked_docs[doc_idx][sent_idx] = tokenizer.mask_token
    src="<doc-sep>".join([" ".join(doc) for doc in masked_docs])
    src=tokenizer(src,max_length=max_len_input,padding="max_length",truncation=True)
    tgt=" ".join(tgt)
    tgt=tokenizer(tgt,max_length=max_len_output,padding="max_length",truncation=True)
    input_ids=src.input_ids
    global_attention_mask=[0 for _ in range(len(input_ids))]
    global_attention_mask[input_ids==tokenizer.vocab["<doc-sep>"]]=1
    global_attention_mask[0]=1
    labels=tgt.input_ids
    labels = [label if label != tokenizer.pad_token_id else -100 for label in labels] 
    return {
        "input_ids":torch.tensor(input_ids,dtype=torch.long),
        "attention_mask":torch.tensor(src.attention_mask,dtype=torch.long),
        "global_attention_mask":torch.tensor(global_attention_mask,dtype=torch.long),
        "labels":torch.tensor(labels,dtype=torch.long)
    }

def process_data(all_docs,tokenizer,scorer,max_len_input=4096,max_len_output=256,mask_ratio=0.2,non_mask_ratio=0.5):
    entity_pyramid=get_entities(all_docs)
    rouge_score_dict=compute_rouge_scores(all_docs,scorer)
    truncated_docs=truncate(all_docs,max_len_input,mask_ratio,non_mask_ratio)
    print("----------------------\n")
    print("Entity pyramid", entity_pyramid)
    print("----------------------\n")
    print("Rouge score dict", rouge_score_dict)
    print("----------------------\n")
    print("truncated_docs:\n",truncated_docs)
    total_num_sentences = sum([len(doc) for doc in truncated_docs])
    num_sents_to_mask = int(total_num_sentences * mask_ratio)
    selected_sents=select_salient_sentences(truncated_docs,rouge_score_dict,entity_pyramid,num_sents_to_mask)
    print("----------------------\n")
    print("Selected sents: ",selected_sents)
    data=get_src_tgt_and_mask(all_docs, selected_sents,tokenizer,max_len_input,max_len_output,non_mask_ratio)
    return data

all_docs = [
    [
        "The global economy is experiencing turbulence due to inflation and market instability.",
        "Stock markets have been highly volatile, with major indexes seeing sharp declines.",
        "Experts warn that economic policies must adapt to avoid a prolonged recession."
    ],
    [
        "A team of astronomers has discovered a new exoplanet that may support life.",
        "The planet, located in the habitable zone,Mars has a stable climate and liquid water.",
        "Some researchers believe space exploration advancements could help mitigate future resource shortages on Earth."
    ],
    [
        "A recent cybersecurity breach targeted financial institutions, exposing sensitive data.",
        "Hackers exploited weaknesses in outdated security systems,Mars leading to major financial losses.",
        "Authorities stress that stronger regulations and AI-driven security measures are needed to prevent future attacks."
    ],
    [
        "In the midst of global uncertainty, the national football team’s victory brought a rare moment of joy.",
        "Fans across Mars the country gathered to celebrate, providing a temporary escape from economic and political challenges.",
        "Analysts suggest that sports events can have a positive psychological impact during difficult times."
    ],
    [
        "Environmental disasters are becoming more frequent due to climate change.",
        "A recent volcanic eruption disrupted air travel and displaced thousands of residents.",
        "Experts warn that rising global temperatures could increase the severity of natural disasters, affecting economies and daily life."
    ]
]

if __name__=="__main__":
    max_len_input=140
    max_len_output=256
    mask_ratio=0.3
    non_mask_ratio=0.5
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL", "rougeLsum"], use_stemmer=True
    )
    # Step 3: Mask the selected sentences
    tokenizer=AutoTokenizer.from_pretrained("allenai/PRIMERA")
    data=process_data(all_docs,tokenizer,scorer,max_len_input,max_len_output)
    #print(truncate(all_docs,max_len_input,mask_ratio,non_mask_ratio))
    print(data)
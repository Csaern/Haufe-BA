import pandas as pd
import random
import pickle
from datasets import Dataset
from tqdm import tqdm
import bm25s
import Stemmer
from multiprocessing import Pool
import gc

df_collection = pd.read_csv("Data/collection.tsv", sep="\t", names=["doc_id", "passage"], on_bad_lines="skip")
df_queries = pd.read_csv("Data/queries.test.tsv", sep="\t", names=["query_id", "query"], on_bad_lines="skip")
df_qrels = pd.read_csv("Data/qrels.test.tsv", sep="\t", names=["query_id", "doc_id"])

with open('Data/models/bm25_passages.pkl', 'rb') as file:
    retriever = pickle.load(file)

stemmer = Stemmer.Stemmer("german")

num_workers = 8

def get_best_bm25_positive(query_tokens, relevant_doc):
    """Berechnet den BM25-Score für jede Passage eines relevanten Dokuments und gibt die beste Passage zurück."""
    passages = df_collection[df_collection['doc_id'] == relevant_doc]['passage'].tolist()

    if not passages:
        return None 

    tokenized_passages = bm25s.tokenize(passages, stopwords="de", stemmer=stemmer)

    bm25_local = bm25s.BM25(k1=1.2, b=0.75)
    bm25_local.index(tokenized_passages)

    results, scores = bm25_local.retrieve(query_tokens, k=1)
    best_results = results.flatten()
    best_passage = passages[best_results[0]]

    del results, scores
    gc.collect()

    return best_passage

def get_hard_negatives(relevant_docs, indices, num_negatives=3):
    """Generiert mehrere harte negative Beispiele aus den Top-k BM25-Ergebnissen."""
    try:
        hard_negative_candidates = []
        for idx in indices.flatten():
            if idx >= len(df_collection):
                continue
            doc_id = df_collection.iloc[idx]['doc_id']
            if doc_id not in relevant_docs:
                hard_negative_candidates.append(df_collection.iloc[idx]['passage'])
        
        if len(hard_negative_candidates) >= num_negatives:
            return random.sample(hard_negative_candidates, num_negatives)
        elif hard_negative_candidates:
            return random.choices(hard_negative_candidates, k=num_negatives)
        else:
            return [None] * num_negatives
    except Exception as e:
        return [None] * num_negatives

def process_query(args):
    query_id, query, relevant_docs_map, all_doc_ids = args
    relevant_docs = relevant_docs_map.get(query_id, set())
    if not relevant_docs:
        return None

    valid_relevant_docs = [doc_id for doc_id in relevant_docs if not df_collection[df_collection['doc_id'] == doc_id]['passage'].empty]
    if not valid_relevant_docs:
        return None

    query_tokens = bm25s.tokenize(query, stopwords="de", stemmer=stemmer)
    
    results = []
    for pos_doc_id in valid_relevant_docs:
        positive = get_best_bm25_positive(query_tokens, pos_doc_id)
        
        indices, scores = retriever.retrieve(query_tokens, k=20)
        hard_negatives = get_hard_negatives(valid_relevant_docs, indices, num_negatives=1)
        
        if positive and all(hard_negatives):
            results.append({
                "anchor": query,
                "positive": positive,
                "hard_negative": hard_negatives[0],
            })

    del query_tokens, indices, scores, valid_relevant_docs, relevant_docs
    gc.collect()
    
    return results

def create_dataset_parallel(queries_df, qrels_df):
    """Erstellt den Datensatz parallel für alle Queries."""
    relevant_docs_map = qrels_df.groupby('query_id')['doc_id'].apply(set).to_dict()
    all_doc_ids = set(df_collection['doc_id'])
    
    queries = [(query_id, query, relevant_docs_map, all_doc_ids) for query_id, query in queries_df.values]
    
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap(process_query, queries), total=len(queries)))

    dataset = {
        "anchor": [],
        "positive": [],
        "hard_negative": [],
        }
    
    for result in results:
        if result is None:
            continue
        for entry in result:
            dataset["anchor"].append(entry["anchor"])
            dataset["positive"].append(entry["positive"])
            dataset["hard_negative"].append(entry["hard_negative"])
    
    return Dataset.from_dict(dataset)

dataset = create_dataset_parallel(df_queries, df_qrels)

print(f"Datensatz: {dataset}")
dataset.save_to_disk("Mxbai/test_dataset_final_v4")
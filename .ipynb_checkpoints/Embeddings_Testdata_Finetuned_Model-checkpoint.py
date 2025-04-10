from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import ndcg_score
import torch
import os
import tqdm
import time
import pickle
from torch.utils.data import DataLoader, Dataset
import multiprocessing as mp
from sklearn.preprocessing import normalize
import gc

model_path = "mixedbread-ai/deepset-mxbai-embed-de-large-v1"
saving_path = "Data/embeddings_v1

"""
Load Datasets, only need Test data for metric evaluation
"""
df_collection = pd.read_csv("Data/collection.tsv", sep="\t", names=["doc_id", "passage"], on_bad_lines="skip")
df_queries = pd.read_csv("Data/queries.test.tsv", sep="\t", names=["query_id", "query"], on_bad_lines="skip")
df_qrels = pd.read_csv("Data/qrels.test.tsv", sep="\t", names=["query_id", "doc_id"])

# Filter collection with documents from qrels_test
relevant_doc_ids = set(df_qrels['doc_id'].unique())
df_collection_filtered = df_collection[df_collection['doc_id'].isin(relevant_doc_ids)]
df_collection_filtered = df_collection_filtered.reset_index(drop=True)

#Load model
model_v1 = SentenceTransformer(f"{model_path}", device="cuda")
batch_size=256

"""
Calculate embeddings
"""

print("Start normalized embeddings")
corpus_embeddings = model.encode(df_collection_filtered["passage"], batch_size=batch_size show_progress_bar=True, normalize_embeddings=True)

# Data Structure for Embedding Data

passage_to_doc_id = dict(enumerate(df_collection_filtered['doc_id']))
embedding_data = {'doc_ids': passage_to_doc_id, 'embeddings': corpus_embeddings}

with open(f'{saving_path}/passages_embeddings_test_only.pkl', 'wb') as file:
    pickle.dump(embedding_data, file)

"""
Calculate query embeddings
"""
query_embeddings = model.encode(df_queries["query"], batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)

# Data Structure for Query Embedding Data
query_id_dict = dict(enumerate(df_queries['query_id']))
query_data = {'query_ids': query_id_dict, 'embeddings': query_embeddings}

"""
Save query embeddings from pickle
"""
with open(f'{saving_path}/query_embeddings_test.pkl', 'wb') as file:
    pickle.dump(query_data, file)


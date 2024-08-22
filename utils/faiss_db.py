import faiss
import numpy as np

def initialize_faiss(embedding_dim):
    index = faiss.IndexFlatL2(embedding_dim)  # L2 distance
    return index

def add_to_faiss(index, embeddings):
    embeddings = np.array(embeddings).astype('float32')
    index.add(embeddings)

def search_faiss(index, query_embedding, top_k=5):
    query_embedding = np.array(query_embedding).astype('float32').reshape(1, -1)
    distances, indices = index.search(query_embedding, top_k)
    return distances, indices

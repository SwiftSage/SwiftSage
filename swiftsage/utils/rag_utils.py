import os
import json

import faiss

from swiftsage.embedding_models import BaseEmbeddingModel


def create_index(embedding_model: BaseEmbeddingModel, retrieval_dataset, dim):
    index = faiss.IndexFlatL2(dim)

    queries = []
    for example in retrieval_dataset:
        query = example['query']
        queries.append(query)
    vectors = embedding_model.batch_encode(queries, dim=dim)
    index.add(vectors)
    return index


def save_index(index, filename):
    faiss.write_index(index, filename)


def load_index(filename):
    return faiss.read_index(filename)


def get_index(embedding_model: BaseEmbeddingModel, retrieval_dataset, dim, index_path):
    if os.path.exists(index_path):
        return load_index(index_path)
    else:
        index = create_index(embedding_model, retrieval_dataset, dim)
        save_index(index, index_path)
        return index

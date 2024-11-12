import numpy as np
import faiss


class RetrievalAugmentation:
    # TODO: implement the retrieval augmentation later
    def __init__(self, embedding_model, retrieval_dataset, embeddings: faiss.IndexFlatL2):
        self.embedding_model = embedding_model
        self.retrieve_dataset = retrieval_dataset
        self.embeddings = embeddings

    def get_similar_examples(self, query, n=3):
        query_embedding = self.get_query_embedding(query)
        _, top_indices = self.index.search(query_embedding, n)
        return [self.retrieve_dataset[i] for i in top_indices]
    
    def get_query_embedding(self, query):
        query_embedding = self.embedding_model.encode(query)
        
        return query_embedding
    
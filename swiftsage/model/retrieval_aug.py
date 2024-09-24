from sklearn.metrics.pairwise import cosine_similarity


class RetrievalAugmentation:
    # TODO: implement the retrieval augmentation later 
    def __init__(self, dataset, embeddings):
        self.dataset = dataset
        self.embeddings = embeddings

    def get_similar_examples(self, query_embedding, n=3):
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        top_indices = similarities.argsort()[-n:][::-1]
        return [self.dataset[i] for i in top_indices]
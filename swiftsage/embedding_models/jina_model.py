import numpy as np
import torch
from transformers import AutoModel

from . import BaseEmbeddingModel


class JinaModel(BaseEmbeddingModel):
    def __init__(self, model_name="jinaai/jina-embeddings-v3"):
        super().__init__(model_name)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).cuda()
        self.model.eval()

    def encode(self, text: str, dim=1024) -> np.ndarray:
        with torch.no_grad():
            embeddings = self.model.encode(text, task="text-matching")
        return embeddings[0]

    def batch_encode(self, texts: list, dim=1024, batch_size=32) -> np.ndarray:
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings.append(self.encode_batch(batch))
        embeddings = np.concatenate(embeddings, axis=0)
        return embeddings

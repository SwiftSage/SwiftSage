import os
import requests

import numpy as np

from . import BaseEmbeddingModel


class JinaAPIModel(BaseEmbeddingModel):
    def __init__(
            self, 
            model_name="jina-embeddings-v3",
            api_base='https://api.jina.ai/v1/embeddings'
        ):
        super().__init__(model_name)
        self.url = api_base
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {os.environ.get("JINA_API_KEY")}'
        }

    def encode(self, text, dim=1024):
        data = {
            "model": self.model_name,
            "task": "text-matching",
            "dimensions": 1024,
            "late_chunking": False,
            "embedding_type": "float",
            "input": [text]
        }
        response = requests.post(self.url, headers=self.headers, json=data)

        res = response.json()
        embedding = np.array(res['data'][0]['embedding']).astype("float32")
        return embedding
    
    def batch_encode(self, texts, dim=1024, batch_size=32):
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            data = {
                "model": self.model_name,
                "task": "text-matching",
                "dimensions": 1024,
                "late_chunking": False,
                "embedding_type": "float",
                "input": batch
            }
            response = requests.post(self.url, headers=self.headers, json=data)

            res = response.json()
            embeddings.append(np.array([r['embedding'] for r in res['data']]).astype("float32"))
        embeddings = np.concatenate(embeddings, axis=0)
        return embeddings
    
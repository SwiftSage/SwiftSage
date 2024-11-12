from abc import ABC, abstractmethod

import numpy as np


class BaseEmbeddingModel(ABC):
    def __init__(self, model_name):
        self.model_name = model_name

    @abstractmethod
    def encode(self, text: str, dim: int) -> np.ndarray:
        pass

    def batch_encode(self, texts: list, dim: int, batch_size: int) -> np.ndarray:
        pass

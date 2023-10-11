import pandas as pd
import numpy as np


def _p2_distance(v1, v2):
    '''
        Returns p2-distance between vectors
    '''
    return np.sqrt(((v1 - v2) ** 2).sum())


def _knn(v, V, k=5, threshold=1.0):
    '''
        Returns indexes of k neirest neighbours
    '''

    distances = [(i, _p2_distance(v, x)) for i, x in enumerate(V)]
    distances = sorted(distances, key=lambda x: x[1])
    indexes = [i for i, d in distances[:k] if d < threshold]

    return indexes


class EmbeddingStorage:
    def __init__(self, path_to_metadata, path_to_embedding):
        # Memorize paths for destructor
        self.path_to_metadata = path_to_metadata
        self.path_to_embedding = path_to_embedding

        self.meta = pd.read_csv(path_to_metadata)
        self.embedding = np.load(path_to_embedding)

    def recoginse_embedding(self, embedding, threshold=1.0):
        '''
            Returns name of recognized person or None
        '''

        indexes = _knn(embedding, self.embeddings)
        if len(indexes) == 0:
            return None

        items = self.meta.iloc[indexes]['name']
        recognized_name = items.mode(axies=1)

        return recognized_name

    def __del__(self):
        self.meta.to_csv(self.path_to_metadata)

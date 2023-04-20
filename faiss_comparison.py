import faiss
import numpy as np
import pandas as pd

class FaissComparison:
    """
    Semantic search using Facebook AI Similarity Search
    Retrieves k-closest embeddings to our query embedding
    """
    def __init__(self, data):
        d = data.values()[0].shape[1]
        self.index = faiss.IndexFlatL2(d)
        df_data = {'id': data.keys(), 'embeddings': data.values()}
        self.df = pd.DataFrame(data = df_data)
        self.embeddings = data.values()
        self.index.add(self.embeddings)

    
    def get_k(self, query_embedding, k:int):
        scores, indices = self.index.search(query_embedding, k) 
        return scores, self.df['id'].iloc[indices]

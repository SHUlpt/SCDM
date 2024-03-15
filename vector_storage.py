import time
import faiss
import numpy as np
from FlagEmbedding import BGEM3FlagModel


class VectorStorage:
    def __init__(self, model_name="BAAI/bge-m3", use_fp16=True, dim=1024):
        self.model = BGEM3FlagModel(model_name, use_fp16=use_fp16)
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.texts = []

    def vectorize(self, text):
        """
        文本向量化
        """
        embedding = self.model.encode(text)["dense_vecs"]
        return embedding

    def add_to_index(self, text):
        """
        添加文本到索引
        """
        vector = self.vectorize(text).astype(np.float32)
        self.index.add(np.array([vector]))
        self.texts.append(text)

    def get_similar_documents(self, query, k=5):
        """
        搜索相似文本
        """
        query_vector = self.vectorize(query)
        scores, indices = self.index.search(
            np.array([query_vector]).astype(np.float32), k
        )
        results = []
        for score, idx in zip(scores[0], indices[0]):
            results.append({"text": self.texts[idx], "score": score})
        return results

    def clear(self):
        """
        清空索引
        """
        self.index.reset()
        self.texts = []
        return "知识库已清空！"

vector_storage = VectorStorage()
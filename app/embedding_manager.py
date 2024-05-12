from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

class EmbeddingManager:
    def __init__(self, model_path, device, normalize_embeddings):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': normalize_embeddings}
            )

    def create_vector_store(self, docs):
        return FAISS.from_documents(docs, self.embeddings)
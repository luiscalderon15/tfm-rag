
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from pathlib import Path

from src.embedding import get_embedding_model


def build_vectorstore(
    documents: list[Document],
    vectorstore_path:Path
) -> FAISS:

    vectorstore_path.parent.mkdir(parents=True, exist_ok=True)

    embeddings = get_embedding_model()


    vectorstore = FAISS.from_documents(documents, embedding=embeddings)

    vectorstore.save_local(vectorstore_path)

    return vectorstore

def load_vectorstore(vectorstore_path:Path)->FAISS:

    vectorstore = FAISS.load_local(
        vectorstore_path,
        get_embedding_model(),
        allow_dangerous_deserialization=True)
    
    return vectorstore
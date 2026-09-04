import torch
from langchain_huggingface import HuggingFaceEmbeddings
 
def get_default_device() -> str:
    """Prefer Apple Silicon's Metal backend (mps) when available, else CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
 
EMBEDDING_MODEL = "BAAI/bge-m3"
 
def get_embedding_model(
    model_name: str = EMBEDDING_MODEL,
    device: str | None = None,
) -> HuggingFaceEmbeddings:
    """
    Returns a LangChain-compatible embedding model.
 
    device=None auto-detects: uses "mps" on Apple Silicon Macs (e.g. M1) if
    available, falling back to CPU otherwise.
 
    normalize_embeddings=True is important: it keeps cosine similarity and dot
    product equivalent, consistent with the equivalence already noted for the
    retrieval step (RRF fusion assumes comparable, normalized similarity scores).
    """
    device = device or get_default_device()
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )
 
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

embedding_model = get_embedding_model()

def comparison(lista:list[str])->np.array:
    vector = embedding_model.embed_documents(lista)
    similarity_matrix = cosine_similarity(vector)
    return similarity_matrix


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_cosine_similarity_heatmap(sim_job) -> None:
    df_similitud = pd.DataFrame(sim_job)

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        df_similitud,
        annot=True,
        cmap="coolwarm",
        vmin=0,
        vmax=1,
        linewidths=0.5,
        square=True,
    )

    plt.title("Mapa de Calor de Similitud de Coseno")
    plt.xlabel("Vectores")
    plt.ylabel("Vectores")
    plt.tight_layout()
    plt.show()
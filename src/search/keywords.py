from rank_bm25 import BM25Okapi
from langchain_community.vectorstores import FAISS

from config import CHUNK_ID_FIELD

def build_bm25_index(vectorstore:FAISS, chunk_id_field:str=CHUNK_ID_FIELD):
  """Build a BM25 index over the same chunk corpus already embedded in the FAISS vectorstore."""
  docs_by_chunk_id = {}
  chunk_ids = []
  tokenized_corpus = []

  for doc in vectorstore.docstore._dict.values():
    chunk_id = doc.metadata.get(chunk_id_field)
    if chunk_id is None:
      continue
    docs_by_chunk_id[chunk_id] = doc
    chunk_ids.append(chunk_id)
    tokenized_corpus.append(doc.page_content.lower().split())

  bm25 = BM25Okapi(tokenized_corpus)
  return docs_by_chunk_id, bm25, chunk_ids
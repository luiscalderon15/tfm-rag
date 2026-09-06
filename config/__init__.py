from pathlib import Path

from config.loader import load_yaml

BASE_DIR = Path(__file__).resolve().parents[1]

MODELS_CONFIG = load_yaml("models.yml")
LLM_PROVIDER = MODELS_CONFIG.get("llm_provider", "llm_opensource")

RETRIEVAL_CONFIG = MODELS_CONFIG.get("retrieval", {})
SEMANTIC_K = RETRIEVAL_CONFIG.get("semantic_k", 30)
KEYWORD_K = RETRIEVAL_CONFIG.get("keyword_k", 30)
RRF_K = RETRIEVAL_CONFIG.get("rrf_k", 60)
SHORTLIST_SIZE = RETRIEVAL_CONFIG.get("shortlist_size", 15)
MAX_EVIDENCE_PER_CANDIDATE = RETRIEVAL_CONFIG.get("max_evidence_per_candidate", 3)
DEFAULT_TOP_N = RETRIEVAL_CONFIG.get("default_top_n", 10)
FACETS_COUNT = RETRIEVAL_CONFIG.get("facets_count", 5)
RETRIEVAL_MODE = RETRIEVAL_CONFIG.get("retrieval_mode", "hybrid")
USE_RERANK = RETRIEVAL_CONFIG.get("use_rerank", False)

_RERANKER_CONFIG = RETRIEVAL_CONFIG.get("reranker", {})
RERANKER_DEFAULT_MODEL = _RERANKER_CONFIG.get("default_model", "BAAI/bge-reranker-v2-m3")
RERANKER_ALTERNATIVE_MODEL = _RERANKER_CONFIG.get("alternative_model", "cross-encoder/ms-marco-MiniLM-L-12-v2")

CANDIDATE_ID_FIELD = RETRIEVAL_CONFIG.get("candidate_id_field", "id")
CHUNK_ID_FIELD = RETRIEVAL_CONFIG.get("chunk_id_field", "id_chunk")

DATA_FOLDER = BASE_DIR / "data"
CVS_FOLDER  = DATA_FOLDER / "samples"

CVS_JUNIOR = CVS_FOLDER / "junior"
CVS_TRAINEE = CVS_FOLDER / "trainee"
CVS_CIRA = CVS_FOLDER / "cira"


VECTOR_STORE_PATH = DATA_FOLDER/"vectorstore/resumes"
VECTOR_DB = DATA_FOLDER/"vectorstore"

VECTOR_STORE_CIRA = VECTOR_DB/"cira"
VECTOR_STORE_JUNIOR = VECTOR_DB/"junior"
VECTOR_STORE_TRAINEE = VECTOR_DB/"trainee"
VECTOR_STORE_FULL = VECTOR_DB/"full-resumes"

if __name__ == "__main__":
    print(BASE_DIR)
import streamlit as st

from config import VECTOR_STORE_TRAINEE,VECTOR_STORE_JUNIOR,VECTOR_STORE_CIRA,VECTOR_STORE_FULL
from src.vectorstore import load_vectorstore
from src.search.hybrid_retriever import HybridCandidateRetriever
from src.bot import evaluate_candidates


st.set_page_config(page_title="Candidate Screening Assistant")
st.title("Candidate Screening Assistant")


VECTORSTORE_OPTIONS = {
  "Junior": VECTOR_STORE_JUNIOR,
  "Trainee": VECTOR_STORE_TRAINEE,
  "Cira": VECTOR_STORE_CIRA,
  "Full": VECTOR_STORE_FULL,
}


@st.cache_resource
def get_retriever(vectorstore_path):
  vectorstore = load_vectorstore(vectorstore_path)
  return HybridCandidateRetriever(vectorstore)


pool_label = st.selectbox("Candidate pool", list(VECTORSTORE_OPTIONS.keys()))
retriever = get_retriever(VECTORSTORE_OPTIONS[pool_label])

jd_full = st.text_area("Job description", height=300)
facets_raw = st.text_area(
  "Facets (optional, one per line — leave empty to use the full JD as a single query)",
  height=150,
)
top_n = st.slider("Number of candidates to shortlist", min_value=1, max_value=20, value=5)
use_rerank = st.checkbox("Use cross-encoder reranking", value=False)

if st.button("Find candidates") and jd_full.strip():
  facets = [line.strip() for line in facets_raw.splitlines() if line.strip()]
  queries = facets if facets else jd_full

  with st.spinner("Searching candidates..."):
    results = retriever.retrieve(queries=queries, rerank_query=jd_full, top_n=top_n, use_rerank=use_rerank)

  if not results:
    st.warning("No matching candidates found.")
  else:
    with st.spinner("Evaluating shortlist..."):
      response = evaluate_candidates(jd_full, results)

    evidence_by_candidate = {result.candidate_id: result.evidence for result in results}

    st.subheader(f"Recommended: {response.recommended_candidate_id}")

    for evaluation in sorted(response.evaluations, key=lambda e: e.rank):
      with st.expander(f"#{evaluation.rank} — {evaluation.candidate_id}"):
        st.markdown(f"**Justification:** {evaluation.justification}")

        if evaluation.matched_requirements:
          st.markdown("**Matched requirements:**")
          for requirement in evaluation.matched_requirements:
            st.markdown(f"- {requirement}")

        if evaluation.gaps:
          st.markdown("**Not evidenced in the data provided (may still apply — not shown as absent):**")
          for gap in evaluation.gaps:
            st.markdown(f"- {gap}")

        st.markdown("**Supporting evidence:**")
        for evidence in evidence_by_candidate.get(evaluation.candidate_id, []):
          st.caption(f"[{evidence.facet}] rerank={evidence.rerank_score}")
          st.write(evidence.chunk_text)

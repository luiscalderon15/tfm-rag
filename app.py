import pandas as pd
import streamlit as st

from config import VECTOR_STORE_TRAINEE,VECTOR_STORE_JUNIOR,VECTOR_STORE_CIRA,VECTOR_STORE_FULL
from src.vectorstore import load_vectorstore
from src.search.hybrid_retriever import HybridCandidateRetriever
from src.bot import draft_outreach_email, evaluate_candidates, generate_candidate_narratives
from src.eval import generate_facets


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
top_n = st.slider("Number of candidates to shortlist", min_value=1, max_value=20, value=5)
use_rerank = st.checkbox("Use cross-encoder reranking", value=False)

if st.button("Find candidates") and jd_full.strip():
  with st.spinner("Extracting facets..."):
    facets = generate_facets(jd_full)

  queries = facets if facets else jd_full

  with st.spinner("Searching candidates..."):
    results = retriever.retrieve(queries=queries, rerank_query=jd_full, top_n=top_n, use_rerank=use_rerank)

  if not results:
    st.warning("No matching candidates found.")
  else:
    with st.spinner("Evaluating shortlist..."):
      response = evaluate_candidates(jd_full, results)

    with st.spinner("Writing candidate summaries..."):
      narratives = generate_candidate_narratives(jd_full, results, response.evaluations)

    st.session_state["screening"] = {
      "job_description": jd_full,
      "facets": facets,
      "results": results,
      "response": response,
      "narratives": narratives,
    }
    # Drafts belong to a previous screening run — drop them so a stale draft from
    # a different job description/candidate set can't be shown against this one.
    for key in list(st.session_state.keys()):
      if key.startswith("email_draft_"):
        del st.session_state[key]

screening = st.session_state.get("screening")

if screening:
  results = screening["results"]
  response = screening["response"]
  narratives = screening["narratives"]

  narrative_by_id = {narrative.candidate_id: narrative for narrative in narratives.narratives}
  retrieval_rank_by_id = {result.candidate_id: position for position, result in enumerate(results, start=1)}
  score_by_id = {result.candidate_id: result.score for result in results}

  with st.expander("Facets used for search", expanded=False):
    for facet in screening["facets"]:
      st.markdown(f"- {facet}")

  st.subheader(f"Recommended: {response.recommended_candidate_id}")

  ranking_table = pd.DataFrame([
    {
      "LLM rank": evaluation.rank,
      "Retrieval rank": retrieval_rank_by_id.get(evaluation.candidate_id),
      "Candidate ID": evaluation.candidate_id,
      "Retrieval score": score_by_id.get(evaluation.candidate_id),
    }
    for evaluation in sorted(response.evaluations, key=lambda e: e.rank)
  ])
  st.dataframe(ranking_table, hide_index=True, use_container_width=True)

  for evaluation in sorted(response.evaluations, key=lambda e: e.rank):
    with st.expander(f"#{evaluation.rank} — {evaluation.candidate_id}"):
      narrative = narrative_by_id.get(evaluation.candidate_id)
      if narrative:
        st.markdown("**Candidate summary**")
        st.markdown(narrative.candidate_summary)
        st.markdown("**Recruiter assessment**")
        st.markdown(narrative.recruiter_assessment)

      draft_key = f"email_draft_{evaluation.candidate_id}"

      if st.button("Draft outreach email", key=f"email_btn_{evaluation.candidate_id}"):
        with st.spinner("Drafting email..."):
          st.session_state[draft_key] = draft_outreach_email(
            screening["job_description"], evaluation, narrative
          )

      draft = st.session_state.get(draft_key)
      if draft:
        st.divider()
        st.caption("Draft only — nothing is sent. Review and copy into your own email client.")
        st.text_input("Subject", value=draft.subject, key=f"subject_{evaluation.candidate_id}")
        st.text_area("Body", value=draft.body, key=f"body_{evaluation.candidate_id}", height=320)

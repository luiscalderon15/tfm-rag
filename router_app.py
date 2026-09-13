import pandas as pd
import streamlit as st

from config import LLM_PROVIDER, MODELS_CONFIG, VECTOR_STORE_TRAINEE, VECTOR_STORE_JUNIOR, VECTOR_STORE_CIRA, VECTOR_STORE_FULL
from src.vectorstore import load_vectorstore
from src.search.hybrid_retriever import HybridCandidateRetriever
from src.router import route


st.set_page_config(page_title="Agentic Router (test)")
st.title("Agentic Router — test")
st.caption(
  "Free-text entry point over the existing tools (screen_candidates, get_candidate, "
  "answer_about_candidate, draft_outreach_email). One tool call per message, no chaining."
)


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

LLM_OPTIONS = [name for name, value in MODELS_CONFIG.items() if name.startswith("llm_") and isinstance(value, dict)]
provider = st.selectbox(
  "LLM provider (defaults to models.yml's llm_provider — pick llm_opensource to test for free)",
  LLM_OPTIONS,
  index=LLM_OPTIONS.index(LLM_PROVIDER) if LLM_PROVIDER in LLM_OPTIONS else 0,
)

if "router_history" not in st.session_state:
  st.session_state.router_history = []
if "current_candidate_id" not in st.session_state:
  st.session_state.current_candidate_id = None

with st.container():
  col1, col2 = st.columns([4, 1])
  with col1:
    if st.session_state.current_candidate_id:
      st.caption(f"Currently discussing: **{st.session_state.current_candidate_id}**")
    else:
      st.caption("No candidate currently active — follow-up questions need a candidate_id.")
  with col2:
    if st.button("Clear candidate", disabled=not st.session_state.current_candidate_id):
      st.session_state.current_candidate_id = None
      st.rerun()


def render_screen_candidates(payload, key_prefix):
  if "error" in payload:
    st.warning(payload["error"])
    return

  results = payload["results"]
  response = payload["response"]
  narratives = payload["narratives"]

  narrative_by_id = {narrative.candidate_id: narrative for narrative in narratives.narratives}
  retrieval_rank_by_id = {result.candidate_id: position for position, result in enumerate(results, start=1)}
  score_by_id = {result.candidate_id: result.score for result in results}

  with st.expander("Facets used for search", expanded=False):
    for facet in payload["facets"]:
      st.markdown(f"- {facet}")

  st.markdown(f"**Recommended: {response.recommended_candidate_id}**")

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


def render_get_candidate(payload, key_prefix):
  if "error" in payload:
    st.warning(payload["error"])
    return

  candidate = payload["candidate"]
  st.markdown(f"**{candidate.candidate_id}**")
  st.write(f"Years of experience: {candidate.years_exp}")
  st.write(f"Skills: {', '.join(candidate.skills) or 'none listed'}")
  st.write(f"Certifications: {', '.join(candidate.certifications) or 'none listed'}")
  if candidate.about_me:
    st.write(f"About me: {candidate.about_me}")


def render_answer_about_candidate(payload, key_prefix):
  if "error" in payload:
    st.warning(payload["error"])
    return

  st.markdown(f"**{payload['candidate_id']}** — _{payload['question']}_")
  st.write(payload["answer"])


def render_draft_outreach_email(payload, key_prefix):
  if "error" in payload:
    st.warning(payload["error"])
    return

  draft = payload["draft"]
  st.markdown(f"Draft for **{payload['candidate_id']}**")
  st.caption("Draft only — nothing is sent.")
  st.text_input("Subject", value=draft.subject, key=f"{key_prefix}_subject")
  st.text_area("Body", value=draft.body, key=f"{key_prefix}_body", height=320)


RENDERERS = {
  "screen_candidates": render_screen_candidates,
  "get_candidate": render_get_candidate,
  "answer_about_candidate": render_answer_about_candidate,
  "draft_outreach_email": render_draft_outreach_email,
}


for index, (user_message, result) in enumerate(st.session_state.router_history):
  with st.chat_message("user"):
    st.write(user_message)

  with st.chat_message("assistant"):
    if result["type"] == "message":
      st.write(result["content"])
    else:
      st.caption(f"tool: {result['tool']} | args: {result['args']}")
      RENDERERS[result["tool"]](result["result"], key_prefix=f"hist_{index}")


message = st.chat_input(
  "e.g. \"Busca candidatos con SQL y Python\", \"Enseñame el perfil de <candidate_id>\", "
  "\"<candidate_id> sabe Docker?\", \"Redacta un email para <candidate_id>...\""
)

if message:
  with st.chat_message("user"):
    st.write(message)

  with st.spinner("Routing..."):
    result = route(
      message, retriever, provider=provider,
      current_candidate_id=st.session_state.current_candidate_id,
    )

  if result.get("resolved_candidate_id"):
    st.session_state.current_candidate_id = result["resolved_candidate_id"]

  with st.chat_message("assistant"):
    if result["type"] == "message":
      st.write(result["content"])
    else:
      st.caption(f"tool: {result['tool']} | args: {result['args']}")
      RENDERERS[result["tool"]](result["result"], key_prefix=f"live_{len(st.session_state.router_history)}")

  st.session_state.router_history.append((message, result))

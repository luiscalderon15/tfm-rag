import streamlit as st

from config import VECTOR_STORE_TRAINEE, VECTOR_STORE_JUNIOR, VECTOR_STORE_CIRA, VECTOR_STORE_FULL
from src.vectorstore import load_vectorstore
from src.search.hybrid_retriever import HybridCandidateRetriever
from src.bot import answer_about_candidate


st.set_page_config(page_title="Candidate Chat")
st.title("Candidate Chat")
st.markdown(
  "Elige el pool, introduce el ID de un candidato (ej. `cv_010`, "
  "`CV_AdrianRodriguez_eng`) y después pregúntale lo que quieras — la "
  "respuesta se basa únicamente en su CV completo, skills, certificaciones "
  "y años de experiencia, sin inventar nada fuera de esos datos."
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

temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
st.caption(
  "Con Ollama siempre se aplica. Con Azure solo si `AZURE_SUPPORTS_TEMPERATURE=true` "
  "en tu `.env` — el modelo actual (o4-mini) no admite temperatura personalizada."
)

candidate_id_input = st.text_input("Candidate ID")

if st.button("Load candidate") and candidate_id_input.strip():
  try:
    candidate = retriever.get_candidate(candidate_id_input.strip())
    st.session_state.candidate = candidate
    st.session_state.messages = []
  except ValueError as e:
    st.session_state.pop("candidate", None)
    st.error(str(e))

if "candidate" in st.session_state:
  candidate = st.session_state.candidate

  st.divider()
  st.subheader(f"Asking about: {candidate.candidate_id}")
  st.caption(
    f"{candidate.years_exp} years — "
    f"Skills: {', '.join(candidate.skills) or 'none listed'}"
  )

  for message in st.session_state.messages:
    with st.chat_message(message["role"]):
      st.markdown(message["content"])

  question = st.chat_input("Ask something about this candidate...")
  if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
      st.markdown(question)

    with st.spinner("Thinking..."):
      answer = answer_about_candidate(question, candidate, temperature=temperature)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
      st.markdown(answer)

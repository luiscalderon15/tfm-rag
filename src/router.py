from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from config import DEFAULT_TOP_N
from src.bot import (
  answer_about_candidate as _answer_about_candidate,
  draft_outreach_email as _draft_outreach_email,
  evaluate_candidates,
  generate_candidate_narratives,
)
from src.eval import generate_facets
from src.llm import get_tool_calling_client

ROUTER_SYSTEM_PROMPT = """You are the router for a recruiting assistant. Given a hiring manager's free-text request, decide which ONE tool (if any) fulfills it, and extract the exact arguments for that tool from the message.

Available actions:
- screen_candidates: rank and evaluate all candidates in the currently selected pool against a job description. Use when the user wants to find, rank, or shortlist candidates for a role.
- get_candidate: look up ONE specific candidate's full profile directly by ID, with no ranking or evaluation. Use when the user names a specific candidate_id and wants to see their info.
- answer_about_candidate: answer a specific free-form question about ONE named candidate. Use when the user asks something about a specific candidate_id (e.g. "does candidate X know Docker?").
- draft_outreach_email: draft an outreach/interview-invitation email for ONE named candidate, given a job description. Use when the user asks to draft or write an email/invitation for a specific candidate_id.

Rules:
- Call AT MOST ONE tool, even if the request seems to need several steps. If it needs more than one action, call only the single most relevant one — the user can ask again for the rest.
- Only extract a candidate_id, job_description, or question that is EXPLICITLY present in the user's message — never invent, guess, or complete one that isn't there.
- candidate_id values are opaque identifiers that often look like "download_FirstName_LastName" or "CV_FirstName_LastName" — copy the ENTIRE token exactly as it appears in the message, byte-for-byte, including any prefix like "download_" or "CV_". Never strip, shorten, "clean up", or otherwise treat part of it as an ordinary word.
- If the request doesn't clearly match any tool, or is missing information a tool needs (e.g. no job description given for screen_candidates, or no candidate_id given for the others), do not call a tool — reply in plain text asking for the missing information instead.
"""


def _build_tools(retriever):
  """
  Wraps the existing, already-validated pipeline functions as LangChain tools,
  closing over `retriever` (bound to one candidate pool) so the LLM only ever
  has to extract candidate_id/job_description/question — never a pool or a
  retriever instance, which aren't things a free-text message could reliably name.
  """

  @tool
  def screen_candidates(job_description: str, top_n: int = DEFAULT_TOP_N) -> dict:
    """Rank and evaluate every candidate in the current pool against a job description. Returns the shortlist, ranking, and per-candidate narrative summaries."""
    facets = generate_facets(job_description)
    results = retriever.retrieve(queries=facets, rerank_query=job_description, top_n=top_n)
    if not results:
      return {"error": "No matching candidates found."}

    response = evaluate_candidates(job_description, results)
    narratives = generate_candidate_narratives(job_description, results, response.evaluations)
    return {
      "job_description": job_description,
      "facets": facets,
      "results": results,
      "response": response,
      "narratives": narratives,
    }

  @tool
  def get_candidate(candidate_id: str) -> dict:
    """Look up ONE candidate directly by their exact candidate_id, bypassing search/ranking. Returns their full anonymized profile."""
    try:
      candidate = retriever.get_candidate(candidate_id)
    except ValueError as error:
      return {"error": str(error)}
    return {"candidate": candidate, "candidate_id": candidate.candidate_id}

  @tool
  def answer_about_candidate(candidate_id: str, question: str) -> dict:
    """Answer a free-form question about ONE specific candidate, identified by their exact candidate_id."""
    try:
      candidate = retriever.get_candidate(candidate_id)
    except ValueError as error:
      return {"error": str(error)}
    answer = _answer_about_candidate(question, candidate)
    return {"candidate_id": candidate_id, "question": question, "answer": answer}

  @tool
  def draft_outreach_email(candidate_id: str, job_description: str) -> dict:
    """Draft a personalized outreach/interview-invitation email for ONE specific candidate, identified by their exact candidate_id, for a given job description."""
    try:
      candidate = retriever.get_candidate(candidate_id)
    except ValueError as error:
      return {"error": str(error)}
    response = evaluate_candidates(job_description, [candidate])
    narratives = generate_candidate_narratives(job_description, [candidate], response.evaluations)
    draft = _draft_outreach_email(job_description, response.evaluations[0], narratives.narratives[0])
    return {"candidate_id": candidate_id, "draft": draft}

  return [screen_candidates, get_candidate, answer_about_candidate, draft_outreach_email]


def _build_system_prompt(current_candidate_id: str = None) -> str:
  if not current_candidate_id:
    return ROUTER_SYSTEM_PROMPT

  return ROUTER_SYSTEM_PROMPT + (
    f"\n\nConversation context: the candidate currently being discussed is "
    f"\"{current_candidate_id}\" (this is a verified fact from this conversation, not "
    f"a guess). If the user's message is a follow-up that clearly continues talking "
    f"about this same candidate without repeating their ID (e.g. \"him\", \"her\", "
    f"\"that candidate\", \"tell me more\", \"what about their experience\"), use this "
    f"candidate_id for the tool call. If the message explicitly names a different "
    f"candidate_id, use that one instead — it takes priority over the context above."
  )


def _extract_resolved_candidate_id(result) -> str:
  if not isinstance(result, dict):
    return None
  if isinstance(result.get("candidate_id"), str):
    return result["candidate_id"]
  candidate = result.get("candidate")
  return getattr(candidate, "candidate_id", None)


def route(message: str, retriever, provider: str = None, current_candidate_id: str = None) -> dict:
  """
  Single-hop router: the LLM sees the full tool menu and picks AT MOST ONE tool
  to call for this message — it never chains multiple tool calls itself, and
  there is no second LLM call to compose a final natural-language answer. The
  caller (UI code) renders the raw result based on which tool ran.

  current_candidate_id: the candidate_id the caller has tracked as "currently
  active" (e.g. the last one a tool resolved), so follow-up questions don't have
  to repeat it. This is passed in as explicit, code-tracked state — never guessed
  by the LLM — and only used when the message doesn't name a different candidate.

  Returns one of:
    {"type": "message", "content": str}
      — no tool matched; `content` is the router's plain-text reply (e.g. asking
        for missing information).
    {"type": "tool_result", "tool": str, "args": dict, "result": ..., "resolved_candidate_id": str | None}
      — the named tool ran; `result` is that tool's raw return value.
      `resolved_candidate_id` is the candidate_id that tool resolved (if any) — the
      caller should store it and pass it back in as `current_candidate_id` on the
      next call, so the "active candidate" carries forward across turns.
  """
  tools = _build_tools(retriever)
  tool_by_name = {t.name: t for t in tools}

  client = get_tool_calling_client(provider=provider)
  llm_with_tools = client.bind_tools(tools)

  response = llm_with_tools.invoke([
    SystemMessage(content=_build_system_prompt(current_candidate_id)),
    HumanMessage(content=message),
  ])

  if not response.tool_calls:
    return {"type": "message", "content": response.content}

  tool_call = response.tool_calls[0]
  selected = tool_by_name[tool_call["name"]]
  result = selected.invoke(tool_call["args"])

  return {
    "type": "tool_result",
    "tool": tool_call["name"],
    "args": tool_call["args"],
    "result": result,
    "resolved_candidate_id": _extract_resolved_candidate_id(result),
  }

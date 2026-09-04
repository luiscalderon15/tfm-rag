from pydantic import BaseModel, Field
import ollama

from src.llm import MODEL_OPENSOURCE_CHAT


class CandidateEvaluation(BaseModel):
  candidate_id: str = Field(
    ...,
    description="The candidate's ID exactly as given in the input, e.g. 'CV_AdrianRodriguez_eng'.",
  )
  rank: int = Field(
    ...,
    description="1 = best match among the candidates provided, 2 = second best, and so on.",
  )
  matched_requirements: list[str] = Field(
    default_factory=list,
    description=(
      "Specific job requirements this candidate's evidence supports, described using "
      "the candidate's own experience as given. Do not invent requirements the "
      "evidence does not actually support."
    ),
  )
  gaps: list[str] = Field(
    default_factory=list,
    description=(
      "Job requirements NOT supported by any of this candidate's provided evidence. "
      "Empty list if the evidence covers everything relevant."
    ),
  )
  justification: str = Field(
    ...,
    description=(
      "A concise explanation of this candidate's rank, referencing only the "
      "experience fragments given below for them."
    ),
  )


class ScreeningResponse(BaseModel):
  recommended_candidate_id: str = Field(
    ..., description="The candidate_id judged as the single best overall match."
  )
  evaluations: list[CandidateEvaluation] = Field(
    ...,
    description="One evaluation per candidate provided, ordered by rank ascending (best first).",
  )


class _LLMCandidateEvaluation(BaseModel):
  """Same shape as CandidateEvaluation, but identifies the candidate by position
  instead of by ID string — the LLM is never given the real candidate_id, so it
  can't mistype or paraphrase it."""

  candidate_index: int = Field(
    ...,
    description="The candidate's number exactly as given in the input, e.g. 2 for 'Candidate #2'.",
  )
  rank: int = Field(
    ...,
    description="1 = best match among the candidates provided, 2 = second best, and so on.",
  )
  matched_requirements: list[str] = Field(
    default_factory=list,
    description=(
      "Specific job requirements this candidate's evidence supports, described using "
      "the candidate's own experience as given. Do not invent requirements the "
      "evidence does not actually support."
    ),
  )
  gaps: list[str] = Field(
    default_factory=list,
    description=(
      "Job requirements NOT supported by any of this candidate's provided evidence. "
      "Empty list if the evidence covers everything relevant."
    ),
  )
  justification: str = Field(
    ...,
    description=(
      "A concise explanation of this candidate's rank, referencing only the "
      "experience fragments given below for them."
    ),
  )


class _LLMScreeningResponse(BaseModel):
  recommended_candidate_index: int = Field(
    ..., description="The candidate_index judged as the single best overall match."
  )
  evaluations: list[_LLMCandidateEvaluation] = Field(
    ...,
    description="One evaluation per candidate provided, ordered by rank ascending (best first).",
  )


SYSTEM_PROMPT = """You are a precise recruiting assistant helping a hiring manager screen candidates for a job opening.

You will be given a job description and a set of candidates. Each candidate is labeled ONLY with a number ("Candidate #1", "Candidate #2", ...), followed by their total years of professional experience, followed by a short list of experience fragments pulled from their CV, each labeled with the job requirement (facet) it was retrieved for.

STRICT GROUNDING — this is the most important rule:
- Base your entire evaluation ONLY on the "Total years of professional experience" figure and the experience fragments given below for that specific candidate. These two things are your ONLY source of truth.
- Do not invent, assume, infer, or complete qualifications, skills, tools, or experience that is not explicitly and literally written in the given fragments — even if the job description mentions them and it seems "likely" or "typical" that a candidate in this field would have them.
- Do NOT use your own general/world knowledge about what a candidate in a similar role "usually" knows or has done. If a specific tool, technology, or requirement named in the job description does not appear verbatim (or as an unmistakable direct paraphrase) in that candidate's given fragments, it is NOT a match.
- Before adding anything to matched_requirements, silently check: "can I point to the exact sentence or phrase in this candidate's fragments that supports this?" If you cannot, do not add it — put the requirement in gaps instead, or omit it if it is out of scope for the evidence you were given.

YEARS OF EXPERIENCE:
- Treat the "Total years of professional experience" number as ground truth — do not re-derive or guess it from the fragments.
- If the job description states a required number/range of years of experience, compare it directly against this number. If the candidate's total is clearly below the required range, this is a real gap on seniority/experience — state it explicitly, even if individual skills otherwise match.

OTHER RULES:
- Refer to candidates ONLY by their number (candidate_index). You are not given any name or ID string for candidates — never invent, guess, or repeat one.
- Rank the candidates relative to each other for this specific job description (1 = best match).
- For each candidate, list which job requirements their given fragments actually support (matched_requirements), and which relevant requirements are NOT supported by any of their given fragments (gaps).
- Do not penalize a candidate for a requirement outside the scope of the fragments you were given — only report gaps you can actually observe from the provided evidence (or from a clear years-of-experience shortfall, per the rule above).
- Return ONLY the JSON object matching the schema, nothing else.
"""


def _format_candidates_block(results) -> str:
  blocks = []
  for index, result in enumerate(results, start=1):
    years_exp_line = (
      f"Total years of professional experience: {result.years_exp}"
      if result.years_exp is not None
      else "Total years of professional experience: unknown"
    )
    evidence_lines = "\n".join(
      f"  [{evidence.facet}] {evidence.chunk_text}" for evidence in result.evidence
    )
    blocks.append(f"Candidate #{index}\n{years_exp_line}\n{evidence_lines}")
  return "\n\n".join(blocks)


def _resolve_candidate_id(index: int, results) -> str:
  if not 1 <= index <= len(results):
    raise ValueError(
      f"LLM returned candidate_index={index}, but only {len(results)} candidates "
      f"were provided (valid range 1-{len(results)})."
    )
  return results[index - 1].candidate_id


def evaluate_candidates(
  job_description: str,
  results,
  model: str = MODEL_OPENSOURCE_CHAT,
) -> ScreeningResponse:
  """
  results: the list of CandidateResult returned by HybridCandidateRetriever.retrieve().
  """
  user_message = (
    f"Job Description:\n{job_description}\n\n"
    f"Candidates:\n{_format_candidates_block(results)}"
  )

  response = ollama.chat(
    model=model,
    messages=[
      {"role": "system", "content": SYSTEM_PROMPT},
      {"role": "user", "content": user_message},
    ],
    format=_LLMScreeningResponse.model_json_schema(),
    options={"temperature": 0},
  )
  llm_response = _LLMScreeningResponse.model_validate_json(response["message"]["content"])

  evaluations = [
    CandidateEvaluation(
      candidate_id=_resolve_candidate_id(evaluation.candidate_index, results),
      rank=evaluation.rank,
      matched_requirements=evaluation.matched_requirements,
      gaps=evaluation.gaps,
      justification=evaluation.justification,
    )
    for evaluation in llm_response.evaluations
  ]

  return ScreeningResponse(
    recommended_candidate_id=_resolve_candidate_id(llm_response.recommended_candidate_index, results),
    evaluations=evaluations,
  )

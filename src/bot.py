from pydantic import BaseModel, Field

from src.llm import chat_structured


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
      "Job requirements for which NO evidence was found in this candidate's provided "
      "data (years of experience, skills, certifications, experience fragments). This "
      "means the data given does not show it — NOT that the candidate lacks it in "
      "reality, since a CV/fragment selection is never a complete record of everything "
      "a candidate knows. Empty list if the data given covers everything relevant."
    ),
  )
  justification: str = Field(
    ...,
    description=(
      "Starts with a single short clause synthesizing who this candidate is "
      "professionally, based ONLY on their 'About me' text if one was given (skip "
      "this opening clause entirely if no 'About me' was given — do not invent one). "
      "Then, in the same paragraph, continues with a concise explanation of this "
      "candidate's rank, referencing only the years of experience, skills, "
      "certifications, and experience fragments given for them — never the 'About "
      "me' text itself as evidence for matched_requirements or gaps."
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
      "Job requirements for which NO evidence was found in this candidate's provided "
      "data (years of experience, skills, certifications, experience fragments). This "
      "means the data given does not show it — NOT that the candidate lacks it in "
      "reality, since a CV/fragment selection is never a complete record of everything "
      "a candidate knows. Empty list if the data given covers everything relevant."
    ),
  )
  justification: str = Field(
    ...,
    description=(
      "Starts with a single short clause synthesizing who this candidate is "
      "professionally, based ONLY on their 'About me' text if one was given (skip "
      "this opening clause entirely if no 'About me' was given — do not invent one). "
      "Then, in the same paragraph, continues with a concise explanation of this "
      "candidate's rank, referencing only the years of experience, skills, "
      "certifications, and experience fragments given for them — never the 'About "
      "me' text itself as evidence for matched_requirements or gaps."
    ),
  )


class _LLMScreeningResponse(BaseModel):
  evaluations: list[_LLMCandidateEvaluation] = Field(
    ...,
    description="One evaluation per candidate provided, ordered by rank ascending (best first).",
  )


SYSTEM_PROMPT = """You are a precise recruiting assistant helping a hiring manager screen candidates for a job opening.

You will be given a job description and a set of candidates. Each candidate is labeled ONLY with a number ("Candidate #1", "Candidate #2", ...), followed by their total years of professional experience, their listed skills, their listed certifications, their own short "About me" self-description (if available), and a short list of experience fragments pulled from their CV, each labeled with the job requirement (facet) it was retrieved for.

STRICT GROUNDING — this is the most important rule:
- Base your entire evaluation ONLY on the "Total years of professional experience" figure, the "Skills" list, the "Certifications" list, and the experience fragments given below for that specific candidate. These are your ONLY sources of truth.
- Do not invent, assume, infer, or complete qualifications, skills, tools, or experience that is not explicitly and literally present in that data — even if the job description mentions them and it seems "likely" or "typical" that a candidate in this field would have them.
- Do NOT use your own general/world knowledge about what a candidate in a similar role "usually" knows or has done. If a specific tool, technology, or requirement named in the job description does not appear verbatim (or as an unmistakable direct match) in that candidate's Skills, Certifications, or fragments, it is NOT a match.
- Before adding anything to matched_requirements, silently check: "can I point to this exact item in this candidate's Skills/Certifications/fragments?" If you cannot, do not add it — put the requirement in gaps instead, or omit it if it is out of scope for the evidence you were given.
- IMPORTANT: "gaps" means "not evidenced in the data given to you" — it does NOT mean "the candidate lacks this in real life". The data you receive is a partial, retrieved slice of a candidate's actual background, never their complete profile. Phrase gaps and justifications accordingly (e.g. "no evidence of X in the data provided" rather than "does not have X" or "lacks X").

YEARS OF EXPERIENCE:
- Treat the "Total years of professional experience" number as ground truth — do not re-derive or guess it from the fragments.
- If the job description states a required number/range of years of experience, compare it directly against this number. If the candidate's total is clearly below the required range, this is a real gap on seniority/experience — state it explicitly, even if individual skills otherwise match.

SKILLS AND CERTIFICATIONS:
- Treat the "Skills" and "Certifications" lists as ground truth, same as years of experience — do not re-derive or guess them from the fragments.
- A job requirement can be satisfied directly by an entry in Skills or Certifications, even if that same tool/technology is never mentioned again in the experience fragments.
- An empty Skills or Certifications list means none were recorded for this candidate — do not treat this as a gap by itself unless the job description explicitly requires something from that list.

ABOUT ME:
- "About me" is the candidate's own self-description — it exists ONLY so you can open the justification with a one-clause synthesis of who this candidate is professionally.
- Never use "About me" as evidence for matched_requirements or gaps — it is self-reported narrative, not a verifiable fact like years of experience, skills, certifications, or an experience fragment.
- If no "About me" was given for a candidate, skip the opening clause entirely — do not invent one.

OTHER RULES:
- Refer to candidates ONLY by their number (candidate_index). You are not given any name or ID string for candidates — never invent, guess, or repeat one.
- Rank the candidates relative to each other for this specific job description. Ranks MUST be exactly 1, 2, 3, ... up to the number of candidates given — the best candidate is rank 1, always. Never start numbering from any value other than 1, and never skip or repeat a number.
- For each candidate, list which job requirements their given data actually supports (matched_requirements), and which relevant requirements have no evidence in their given data (gaps).
- Do not penalize a candidate for a requirement outside the scope of the data you were given — only report gaps you can actually observe from what was provided (or from a clear years-of-experience shortfall, per the rule above).

SELF-CONSISTENCY CHECK — do this silently before returning your answer:
- No specific technology/tool/skill may appear in BOTH matched_requirements and gaps for the same candidate. If you find one that does, remove it from gaps.
- Job description requirements are often compound sentences bundling several distinct skills together (e.g. "SQL, Python, Git and Docker; practical experience with Azure..."). Never copy such a bundled sentence into gaps as a whole if you already matched part of it — split it, and put ONLY the specific sub-part you found no evidence for into gaps, worded narrowly (e.g. "SQL" or "Azure Data Factory", not the entire original sentence).
- Your `justification` text must not contradict matched_requirements or gaps — never write "no evidence of X" in the justification if X is listed in matched_requirements, and never claim a match in the justification for something listed in gaps.
- Return ONLY the JSON object matching the schema, nothing else.
"""


def _format_list_field(label: str, values: list) -> str:
  cleaned = [str(v).strip() for v in values if v and str(v).strip() not in ('""', "''")]
  if not cleaned:
    return f"{label}: none listed"
  return f"{label}: " + ", ".join(cleaned)


def _format_about_me(about_me: str) -> str:
  if not about_me or about_me.strip() in ('""', "''"):
    return "About me: not given"
  return f"About me: {about_me.strip()}"


def _format_candidates_block(results) -> str:
  blocks = []
  for index, result in enumerate(results, start=1):
    years_exp_line = (
      f"Total years of professional experience: {result.years_exp}"
      if result.years_exp is not None
      else "Total years of professional experience: unknown"
    )
    skills_line = _format_list_field("Skills", result.skills)
    certifications_line = _format_list_field("Certifications", result.certifications)
    about_me_line = _format_about_me(result.about_me)
    evidence_lines = "\n".join(
      f"  [{evidence.facet}] {evidence.chunk_text}" for evidence in result.evidence
    )
    blocks.append(
      f"Candidate #{index}\n{years_exp_line}\n{skills_line}\n{certifications_line}\n{about_me_line}\n{evidence_lines}"
    )
  return "\n\n".join(blocks)


def _resolve_candidate_id(index: int, results) -> str:
  if not 1 <= index <= len(results):
    raise ValueError(
      f"LLM returned candidate_index={index}, but only {len(results)} candidates "
      f"were provided (valid range 1-{len(results)})."
    )
  return results[index - 1].candidate_id


def _normalize_ranks(evaluations) -> None:
  """
  Re-numbers ranks to a clean 1..N sequence, preserving relative order — fixes
  the common case where the LLM's ranks are valid relative to each other but
  offset or gapped (e.g. it returns [3, 4, 5] instead of [1, 2, 3]). Only
  raises if ranks aren't distinct, since that's genuinely ambiguous to resolve
  automatically (no way to know which candidate should end up where).
  """
  ranks = [evaluation.rank for evaluation in evaluations]
  if len(set(ranks)) != len(ranks):
    raise ValueError(f"LLM returned duplicate ranks: {sorted(ranks)}.")

  for new_rank, evaluation in enumerate(sorted(evaluations, key=lambda e: e.rank), start=1):
    evaluation.rank = new_rank


def evaluate_candidates(
  job_description: str,
  results,
  provider: str = None,
  temperature: float = 0,
) -> ScreeningResponse:
  """
  results: the list of CandidateResult returned by HybridCandidateRetriever.retrieve().
  provider: overrides the configured LLM_PROVIDER for this call only ("ollama" or "azure").
  temperature: see chat_structured — ignored for Azure unless AZURE_SUPPORTS_TEMPERATURE=true.
  """
  user_message = (
    f"Job Description:\n{job_description}\n\n"
    f"Candidates:\n{_format_candidates_block(results)}"
  )

  llm_response = chat_structured(
    system_prompt=SYSTEM_PROMPT,
    user_message=user_message,
    schema=_LLMScreeningResponse,
    provider=provider,
    temperature=temperature,
  )

  if not llm_response.evaluations:
    raise ValueError("LLM returned no evaluations.")

  _normalize_ranks(llm_response.evaluations)

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

  # Derived from the evaluation ranked #1, instead of asking the LLM to name the
  # winner a second time in a separate field — two independent judgments that can
  # (and did) disagree with each other.
  best_evaluation = min(evaluations, key=lambda e: e.rank)

  return ScreeningResponse(
    recommended_candidate_id=best_evaluation.candidate_id,
    evaluations=evaluations,
  )


class CandidateAnswer(BaseModel):
  answer: str = Field(
    ...,
    description=(
      "A direct, natural-language answer to the user's question about this ONE "
      "candidate, grounded only in their years of experience, skills, "
      "certifications, and CV text given below. If the given data does not "
      "address the question, say so plainly instead of guessing or inferring."
    ),
  )


CANDIDATE_QA_SYSTEM_PROMPT = """You are a precise recruiting assistant answering a hiring manager's question about ONE specific candidate.

You will be given that candidate's total years of professional experience, their listed skills, their listed certifications, their own short "About me" self-description (if available), and their CV text.

STYLE:
- Answer the question directly and naturally, as you would speak to a hiring manager — do NOT use a rigid "matched requirements / gaps" checklist format, and do not restate boilerplate (like years of experience) unless it's actually relevant to what was asked.
- Be concise. Only include what's relevant to the actual question.
- Return ONLY the JSON object matching the schema, nothing else.
"""


def answer_about_candidate(question: str, candidate, provider: str = None, temperature: float = 0) -> str:
  """
  Answers a free-form question about ONE candidate directly — no ranking, no
  matched_requirements/gaps, just a grounded natural-language answer. For a
  chatbot that already knows which candidate it's asking about (e.g. via
  HybridCandidateRetriever.get_candidate()).

  temperature: see chat_structured — ignored for Azure unless AZURE_SUPPORTS_TEMPERATURE=true.
  """
  user_message = f"Question: {question}\n\n{_format_candidates_block([candidate])}"

  result = chat_structured(
    system_prompt=CANDIDATE_QA_SYSTEM_PROMPT,
    user_message=user_message,
    schema=CandidateAnswer,
    provider=provider,
    temperature=temperature,
  )
  return result.answer

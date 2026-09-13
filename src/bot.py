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
      "A concise explanation of this candidate's rank, referencing only the "
      "years of experience, skills, certifications, and experience fragments "
      "given for them."
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
      "A concise explanation of this candidate's rank, referencing only the "
      "years of experience, skills, certifications, and experience fragments "
      "given for them."
    ),
  )


class _LLMScreeningResponse(BaseModel):
  evaluations: list[_LLMCandidateEvaluation] = Field(
    ...,
    description="One evaluation per candidate provided, ordered by rank ascending (best first).",
  )


SYSTEM_PROMPT = """You are a precise recruiting assistant helping a hiring manager screen candidates for a job opening.

You will be given a job description and a set of candidates. Each candidate is labeled ONLY with a number ("Candidate #1", "Candidate #2", ...), followed by their total years of professional experience, their listed skills, their listed certifications, and a short list of experience fragments pulled from their CV, each labeled with the job requirement (facet) it was retrieved for.

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


def _format_candidates_block(results, include_about_me: bool = True) -> str:
  blocks = []
  for index, result in enumerate(results, start=1):
    years_exp_line = (
      f"Total years of professional experience: {result.years_exp}"
      if result.years_exp is not None
      else "Total years of professional experience: unknown"
    )
    skills_line = _format_list_field("Skills", result.skills)
    certifications_line = _format_list_field("Certifications", result.certifications)
    evidence_lines = "\n".join(
      f"  [{evidence.facet}] {evidence.chunk_text}" for evidence in result.evidence
    )
    lines = [f"Candidate #{index}", years_exp_line, skills_line, certifications_line]
    if include_about_me:
      lines.append(_format_about_me(result.about_me))
    lines.append(evidence_lines)
    blocks.append("\n".join(lines))
  return "\n\n".join(blocks)


def _format_narrative_candidates_block(results, evaluations) -> str:
  evaluation_by_id = {evaluation.candidate_id: evaluation for evaluation in evaluations}
  blocks = []
  for index, result in enumerate(results, start=1):
    evaluation = evaluation_by_id[result.candidate_id]
    matched_lines = (
      "\n".join(f"  - {requirement}" for requirement in evaluation.matched_requirements)
      or "  none"
    )
    gaps_lines = "\n".join(f"  - {gap}" for gap in evaluation.gaps) or "  none"
    evidence_lines = "\n".join(
      f"  [{evidence.facet}] {evidence.chunk_text}" for evidence in result.evidence
    )
    lines = [
      f"Candidate #{index}",
      _format_about_me(result.about_me),
      "Matched requirements:",
      matched_lines,
      "Not evidenced in the data provided:",
      gaps_lines,
      "Supporting evidence:",
      evidence_lines,
    ]
    blocks.append("\n".join(lines))
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
    f"Candidates:\n{_format_candidates_block(results, include_about_me=False)}"
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


class CandidateNarrative(BaseModel):
  candidate_id: str = Field(
    ..., description="The candidate's ID exactly as given in the input."
  )
  candidate_summary: str = Field(
    ...,
    description=(
      "A short, HR-recruiter-style synthesis of who this candidate is, based only "
      "on their 'About me' self-description."
    ),
  )
  assessment_fit: str = Field(
    ...,
    description=(
      "A short narrative (not a bullet list) connecting this candidate's matched "
      "requirements and supporting evidence to the job description's responsibilities — "
      "why they fit. Grounded only in matched_requirements/evidence already established "
      "— never in 'About me'."
    ),
  )
  assessment_open_questions: str = Field(
    ...,
    description=(
      "A short narrative (not a bullet list) covering this candidate's not-evidenced "
      "requirements (gaps), phrased as open questions rather than definitive "
      "shortcomings. Grounded only in the gaps already established — never in 'About me'."
    ),
  )

  @property
  def recruiter_assessment(self) -> str:
    return f"{self.assessment_fit}\n\n{self.assessment_open_questions}"


class NarrativeResponse(BaseModel):
  narratives: list[CandidateNarrative] = Field(
    ..., description="One narrative pair per candidate provided."
  )


class _LLMCandidateNarrative(BaseModel):
  """Same shape as CandidateNarrative, but identifies the candidate by position —
  see _LLMCandidateEvaluation for why."""

  candidate_index: int = Field(
    ...,
    description="The candidate's number exactly as given in the input, e.g. 2 for 'Candidate #2'.",
  )
  candidate_summary: str = Field(
    ...,
    description=(
      "A short, HR-recruiter-style synthesis of who this candidate is, based only "
      "on their 'About me' self-description."
    ),
  )
  assessment_fit: str = Field(
    ...,
    description=(
      "A short narrative (not a bullet list) connecting this candidate's matched "
      "requirements and supporting evidence to the job description's responsibilities — "
      "why they fit. Grounded only in matched_requirements/evidence already established "
      "— never in 'About me'."
    ),
  )
  assessment_open_questions: str = Field(
    ...,
    description=(
      "A short narrative (not a bullet list) covering this candidate's not-evidenced "
      "requirements (gaps), phrased as open questions rather than definitive "
      "shortcomings. Grounded only in the gaps already established — never in 'About me'."
    ),
  )


class _LLMNarrativeResponse(BaseModel):
  narratives: list[_LLMCandidateNarrative] = Field(
    ..., description="One narrative pair per candidate provided."
  )


NARRATIVE_SYSTEM_PROMPT = """You are a precise recruiting assistant writing a narrative summary for a hiring manager, for each candidate in a shortlist that has already been ranked and evaluated.

You will be given a job description and, for each candidate (labeled "Candidate #1", "Candidate #2", ...): their own "About me" self-description, the job requirements already confirmed as matched (Matched requirements), the requirements not evidenced in their data (Not evidenced in the data provided), and the supporting evidence fragments used to reach those conclusions.

For each candidate, write three short pieces of text, each its own separate paragraph — never combine them into one block:

1. CANDIDATE SUMMARY (candidate_summary):
- A short, natural, HR-recruiter-style synthesis of who this candidate is, based ONLY on their "About me" text.
- If "About me" is "not given" or empty, say so briefly instead of inventing one (e.g. "No self-description was provided for this candidate.").
- Do not use "About me" content anywhere else in your output — it belongs only in this section.

2. FIT (assessment_fit):
- A narrative (NOT a bullet list) that connects the candidate's matched requirements and supporting evidence to the responsibilities/requirements in the job description — why they fit.
- Base this STRICTLY on the given Matched requirements and Supporting evidence — never on "About me" or on the Not evidenced list.

3. OPEN QUESTIONS (assessment_open_questions):
- A narrative (NOT a bullet list) covering the candidate's not-evidenced requirements, phrased as open questions rather than definitive shortcomings (e.g. "no evidence of X in the data reviewed" rather than "lacks X" or "does not have X") — consistent with the fact that this is a partial, retrieved slice of the candidate's real background.
- Base this STRICTLY on the given Not evidenced list — never on "About me" or on the Matched requirements list. If Not evidenced is empty, say plainly that no open questions were identified from the data reviewed.

Do not introduce any requirement, skill, or fact in FIT or OPEN QUESTIONS that is not already present in Matched requirements, Not evidenced, or Supporting evidence — these sections do not re-evaluate the candidate, they only explain the evaluation already made.

SELF-CONSISTENCY CHECK — do this silently before returning your answer:
- Never write "no evidence of X" or otherwise claim X is missing (in assessment_open_questions) if X (or an unmistakable direct match for it) is already listed in Matched requirements — that already means evidence for X was found, and belongs in assessment_fit instead.
- Never claim a match (in assessment_fit) for something that is listed in Not evidenced.
- If you catch yourself about to make either mistake, drop that claim rather than writing it.

STYLE:
- Write in full sentences, not bullet points, for all three fields.
- Be concise — a few sentences per field is enough.
- Refer to candidates ONLY by their number (candidate_index), exactly as given — never invent, guess, or repeat an ID.
- Return ONLY the JSON object matching the schema, nothing else.
"""


def generate_candidate_narratives(
  job_description: str,
  results,
  evaluations: list[CandidateEvaluation],
  provider: str = None,
  temperature: float = 0,
) -> NarrativeResponse:
  """
  Second-stage call, chained after evaluate_candidates: turns its already-validated
  matched_requirements/gaps into a narrative summary and recruiter assessment per
  candidate, batched in a single LLM call. about_me is introduced here for the first
  time and only for candidate_summary — recruiter_assessment must never use it.

  results: the same CandidateResult list passed to evaluate_candidates.
  evaluations: evaluate_candidates(...).evaluations (or ScreeningResponse.evaluations).
  provider/temperature: see chat_structured.
  """
  user_message = (
    f"Job Description:\n{job_description}\n\n"
    f"Candidates:\n{_format_narrative_candidates_block(results, evaluations)}"
  )

  llm_response = chat_structured(
    system_prompt=NARRATIVE_SYSTEM_PROMPT,
    user_message=user_message,
    schema=_LLMNarrativeResponse,
    provider=provider,
    temperature=temperature,
  )

  if not llm_response.narratives:
    raise ValueError("LLM returned no narratives.")

  narratives = [
    CandidateNarrative(
      candidate_id=_resolve_candidate_id(narrative.candidate_index, results),
      candidate_summary=narrative.candidate_summary,
      assessment_fit=narrative.assessment_fit,
      assessment_open_questions=narrative.assessment_open_questions,
    )
    for narrative in llm_response.narratives
  ]

  return NarrativeResponse(narratives=narratives)


class EmailDraft(BaseModel):
  subject: str = Field(..., description="A short, natural subject line for this outreach email.")
  body: str = Field(
    ...,
    description=(
      "The full email body, following the required template structure exactly. "
      "Fill in '[relevant skill/area]' using only the matched requirements given. "
      "Fill in '[Job Title]' and '[Company Name]' only if explicitly stated in the "
      "job description given — otherwise leave those exact bracketed placeholders "
      "untouched. Leave every other bracketed placeholder (candidate's first name, "
      "the call duration choice, all Date & Time options, and the recruiter's name) "
      "exactly as written in the template, unedited."
    ),
  )


EMAIL_TEMPLATE = """Hi [Candidate's First Name],

I came across your profile and was impressed by your experience in [relevant skill/area]. I believe your background could be a great fit for the [Job Title] position at [Company Name].

I’d love to schedule a brief [15/20/30]-minute call to learn more about your experience, tell you a little more about the role, and see whether it could be a good fit for both sides.

Would you be available for a quick conversation at any of the following times?

• [Date & Time]
• [Date & Time]
• [Date & Time]

If none of these work for you, feel free to suggest another time that suits you.

Looking forward to hearing from you!

Best regards,
[Recruiter's Name]
[Job Title]
[Company Name]"""


EMAIL_SYSTEM_PROMPT = """You are a precise recruiting assistant drafting a short outreach email to invite ONE candidate to a screening call, for a hiring manager to review and send themselves — you never send anything yourself.

You MUST follow this exact template — do not rewrite, reorder, paraphrase, or add/remove any part of it:

---
""" + EMAIL_TEMPLATE + """
---

You will be given the job description and, for the one candidate being contacted, their matched requirements and a short recruiter assessment of their fit (already validated in a prior step — do not re-evaluate the candidate, only use this to fill the template).

STRICT RULES:
- Fill in "[relevant skill/area]" with a short, specific reference to one or two of the candidate's actual matched requirements/strengths given below — never invent a skill that isn't listed there.
- Fill in "[Job Title]" and "[Company Name]" ONLY if that exact information is explicitly stated in the job description given below. If either is not explicitly present, leave that exact bracketed placeholder untouched, verbatim (keep writing "[Job Title]" or "[Company Name]" literally).
- Leave every other bracketed placeholder EXACTLY as written in the template, verbatim, with no changes: "[Candidate's First Name]", "[15/20/30]", all three "[Date & Time]" bullets, and "[Recruiter's Name]". You are never given the candidate's real name, the recruiter's identity, or actual scheduling — never invent, guess, or remove these placeholders.
- Do not add any new sentence, section, or signature line beyond the template.
- Return ONLY the JSON object matching the schema, nothing else.
"""


def draft_outreach_email(
  job_description: str,
  evaluation: CandidateEvaluation,
  narrative: CandidateNarrative,
  provider: str = None,
  temperature: float = 0,
) -> EmailDraft:
  """
  Drafts a personalized outreach email (subject + body) for ONE already-ranked
  candidate, following a fixed template. Text-only — never sent automatically,
  only shown to the hiring manager to review/edit/send themselves. Grounded only
  in this candidate's already-validated matched_requirements and recruiter
  assessment (assessment_fit) — never invents a skill, company name, job title,
  candidate name, schedule, or recruiter identity that wasn't explicitly given.
  """
  matched_lines = (
    "\n".join(f"- {requirement}" for requirement in evaluation.matched_requirements)
    or "- none"
  )
  user_message = (
    f"Job Description:\n{job_description}\n\n"
    f"Candidate's matched requirements:\n{matched_lines}\n\n"
    f"Candidate's recruiter assessment (fit):\n{narrative.assessment_fit}"
  )

  return chat_structured(
    system_prompt=EMAIL_SYSTEM_PROMPT,
    user_message=user_message,
    schema=EmailDraft,
    provider=provider,
    temperature=temperature,
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
- Ground your answer ONLY in the information provided above — never invent or infer details that aren't there. If the provided information isn't enough to answer the question, say plainly that you don't know / it isn't stated in the candidate's data, instead of guessing.
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

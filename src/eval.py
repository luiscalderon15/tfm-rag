import json
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field

from config import CANDIDATE_ID_FIELD, RERANKER_ALTERNATIVE_MODEL, RERANKER_DEFAULT_MODEL
from src.llm import chat_structured

FRAGMENTS_FIELD = "anonimized_fragments"


def get_candidate_text(
  df: pd.DataFrame,
  candidate_id: str,
  id_col: str = CANDIDATE_ID_FIELD,
  fragments_col: str = FRAGMENTS_FIELD,
) -> str:
  """
  Fetches one candidate's anonymized fragments from a DataFrame (one row per
  candidate) and joins them into a single text block. `fragments_col` may hold
  either a list of fragment strings or a single already-joined string.
  """
  matches = df.loc[df[id_col] == candidate_id, fragments_col]
  if matches.empty:
    raise ValueError(f"No row found for candidate_id={candidate_id!r} in column {id_col!r}.")

  fragments = matches.iloc[0]
  if isinstance(fragments, (list, tuple, pd.Series)):
    return "\n\n".join(str(f) for f in fragments)
  return str(fragments)


class SyntheticJD(BaseModel):
  job_description: str = Field(
    ...,
    description=(
      "A realistic, self-contained job description in job-posting language, "
      "generalized from the candidate's CV without copying exact wording or "
      "naming real companies/people."
    ),
  )


SYNTHETIC_JD_PROMPT = """You are simulating a hiring manager writing a job posting.

You will be given one candidate's full CV. Write a realistic job description for a
role that this exact candidate would be an excellent match for — someone reading
only the job description (not the CV) should conclude this candidate is a
near-perfect fit.

Critical rules to keep this a fair, realistic test (do not skip these):
- Write the job description the way a real hiring manager would — in generic,
  industry-standard job-posting language. Do NOT copy exact sentences, phrases, or
  verb choices from the CV verbatim; paraphrase and generalize the underlying
  skills/responsibilities instead.
- Do NOT mention any specific company name, product name, client name, or project
  codename from the CV — a hiring manager writing a NEW job posting would never
  know these.
- Do NOT include any personal information: no candidate name, no dates, no contact
  details, no anonymization placeholders (e.g. <PERSON>, <ORGANIZATION>,
  <DATE_TIME>) if any appear in the input — ignore them entirely.
- Include only: role responsibilities, required technical skills/tools/technologies,
  and a reasonable required years-of-experience range, inferred from the seniority
  the CV implies — do not state an exact number pulled directly from the text if
  one appears there.
- The job description must be self-contained and realistic on its own — it must
  not read like a summary of a resume.
- Do not add requirements that this candidate would NOT plausibly meet.

Candidate CV:
{candidate_text}
"""


def generate_synthetic_jd(
  candidate_id: str,
  df: pd.DataFrame,
  id_col: str = CANDIDATE_ID_FIELD,
  fragments_col: str = FRAGMENTS_FIELD,
  provider: str = None,
) -> str:
  """
  Generates a synthetic job description for which `candidate_id` is the intended
  ground-truth best match — for building a retrieval eval set without manual labeling.

  df: one row per candidate, with an id column and a column holding that
  candidate's anonymized fragments (list of strings, or a single joined string).
  """
  candidate_text = get_candidate_text(df, candidate_id, id_col, fragments_col)

  result = chat_structured(
    system_prompt=SYNTHETIC_JD_PROMPT.format(candidate_text=candidate_text),
    user_message="Write the job description now.",
    schema=SyntheticJD,
    provider=provider,
  )
  return result.job_description


def export_synthetic_jds(
  df: pd.DataFrame,
  output_dir: str | Path,
  id_col: str = CANDIDATE_ID_FIELD,
  fragments_col: str = FRAGMENTS_FIELD,
  provider: str = None,
  skip_existing: bool = True,
) -> None:
  """
  Generates a synthetic JD for every candidate in `df` and writes one JSON file per
  candidate to `output_dir`, named `<candidate_id>.json`, with `candidate_id` (the
  ground-truth match for that JD) and `job_description`.

  skip_existing: if True (default), candidates whose JSON already exists in
  output_dir are skipped instead of re-generated — avoids re-paying for LLM calls
  (especially relevant for provider="azure") when resuming a partial run.
  """
  output_dir = Path(output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  candidate_ids = df[id_col].unique()
  for candidate_id in candidate_ids:
    out_path = output_dir / f"{candidate_id}.json"
    if skip_existing and out_path.exists():
      print(f"[skip] {candidate_id} (ya existe {out_path.name})")
      continue

    job_description = generate_synthetic_jd(candidate_id, df, id_col, fragments_col, provider)
    payload = {"candidate_id": candidate_id, "job_description": job_description}
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] {candidate_id} -> {out_path.name}")


# --- Alternative ground truth: PARTIAL match, to actually stress-test facets ---
#
# SYNTHETIC_JD_PROMPT above builds a JD the candidate matches almost perfectly,
# which structurally favors comparing the whole JD against the whole candidate —
# not the case facets are meant for (rewarding coverage across several distinct
# requirements even when no single one is a 100% match). This second prompt
# deliberately mixes in requirements the candidate does NOT satisfy, so the
# ground-truth candidate is the best available match, not a perfect one.

PARTIAL_MATCH_JD_PROMPT = """You are simulating a hiring manager writing a job posting.

You will be given one candidate's full CV. Write a realistic job description for a
role that this candidate would be a GOOD match for, but NOT a perfect one — the JD
must mix requirements this candidate clearly satisfies with a smaller number of
additional, plausible requirements that this candidate's CV does NOT support. The
goal is to simulate a realistic hiring scenario, where even the best available
candidate does not tick every single box.

Structure:
- Roughly 60-70% of the requirements/responsibilities in the JD must be clearly
  grounded in this candidate's actual CV (paraphrased, not copied verbatim).
- The remaining 30-40% must be plausible, realistic requirements for the SAME kind
  of role, but that this candidate's CV does NOT mention or support. These must
  still fit naturally into a single, coherent job posting — do not make them look
  out of place, unrelated, or absurd (e.g., do not mix a data role with a
  bricklaying requirement). Choose gaps a real hiring manager for this kind of role
  would plausibly ask for (an adjacent tool, a nearby seniority expectation, a
  related but not-quite-covered skill).
- Even with these gaps, this candidate should still read as a strong, worthwhile
  applicant for the role overall — not a bad fit, just not a 100% fit.

Critical rules (same as always, do not skip):
- Do NOT copy exact sentences, phrases, or verb choices from the CV verbatim;
  paraphrase and generalize the underlying skills/responsibilities.
- Do NOT mention any specific company name, product name, client name, or project
  codename from the CV.
- Do NOT include any personal information: no candidate name, no dates, no contact
  details, no anonymization placeholders (e.g. <PERSON>, <ORGANIZATION>,
  <DATE_TIME>) if any appear in the input — ignore them entirely.
- Do not state an exact years-of-experience number pulled directly from the text;
  infer a reasonable range instead.
- The job description must be self-contained and realistic on its own.

Candidate CV:
{candidate_text}
"""


def generate_partial_match_jd(
  candidate_id: str,
  df: pd.DataFrame,
  id_col: str = CANDIDATE_ID_FIELD,
  fragments_col: str = FRAGMENTS_FIELD,
  provider: str = None,
) -> str:
  """
  Same idea as generate_synthetic_jd, but the JD deliberately mixes in requirements
  the candidate does NOT satisfy — a partial-match ground truth, meant to test
  whether facet-based retrieval (which rewards coverage across several distinct
  requirements) actually beats a single whole-JD query when the match isn't total.
  """
  candidate_text = get_candidate_text(df, candidate_id, id_col, fragments_col)

  result = chat_structured(
    system_prompt=PARTIAL_MATCH_JD_PROMPT.format(candidate_text=candidate_text),
    user_message="Write the job description now.",
    schema=SyntheticJD,
    provider=provider,
  )
  return result.job_description


def export_partial_match_jds(
  df: pd.DataFrame,
  output_dir: str | Path,
  id_col: str = CANDIDATE_ID_FIELD,
  fragments_col: str = FRAGMENTS_FIELD,
  provider: str = None,
  skip_existing: bool = True,
) -> None:
  """Same as export_synthetic_jds, but using generate_partial_match_jd."""
  output_dir = Path(output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  candidate_ids = df[id_col].unique()
  for candidate_id in candidate_ids:
    out_path = output_dir / f"{candidate_id}.json"
    if skip_existing and out_path.exists():
      print(f"[skip] {candidate_id} (ya existe {out_path.name})")
      continue

    job_description = generate_partial_match_jd(candidate_id, df, id_col, fragments_col, provider)
    payload = {"candidate_id": candidate_id, "job_description": job_description}
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] {candidate_id} -> {out_path.name}")


def recall_at_k(ranked_ids: list, ground_truth_id: str, k: int) -> float:
  """1.0 if ground_truth_id appears within the first k items of ranked_ids, else 0.0."""
  return 1.0 if ground_truth_id in ranked_ids[:k] else 0.0


def mrr(ranked_ids: list, ground_truth_id: str) -> float:
  """Reciprocal rank (1-indexed) of ground_truth_id in ranked_ids, or 0.0 if absent."""
  for rank, candidate_id in enumerate(ranked_ids, start=1):
    if candidate_id == ground_truth_id:
      return 1.0 / rank
  return 0.0


def load_test_set(jds_dir: str | Path) -> list[dict]:
  """Loads every synthetic-JD JSON file from jds_dir into a list of
  {"candidate_id": ..., "job_description": ...} dicts (one per file, as written
  by export_synthetic_jds)."""
  jds_dir = Path(jds_dir)
  return [
    json.loads(path.read_text(encoding="utf-8"))
    for path in sorted(jds_dir.glob("*.json"))
  ]


def evaluate_retrieval(
  test_set: list,
  retriever,
  k_values: list = (1, 3, 5, 10),
  top_n: int = None,
  query_field: str = "job_description",
  **retrieve_kwargs,
) -> dict:
  """
  Runs `retriever` over every (job_description, candidate_id) pair in test_set and
  reports mean Recall@k (per k in k_values) and MRR averaged across the whole set.

  query_field: which field of each test-set item to use as the search query — the
  default "job_description" (the JD as a single query) or "facets" (a list of
  facets, if add_facets_to_synthetic_jds already populated them). The rerank query
  is always the full job_description, regardless of query_field, matching how the
  app always reranks against the full JD even when facets drive retrieval.

  retrieve_kwargs are forwarded to retriever.retrieve(), e.g. use_rerank=False,
  retrieval_mode="semantic", or cross_encoder=<model> to A/B test a pipeline
  variant against the same test set.
  """
  top_n = top_n or max(k_values)

  recalls = {k: [] for k in k_values}
  mrrs = []

  for item in test_set:
    queries = item[query_field]
    job_description = item["job_description"]
    ground_truth_id = item["candidate_id"]

    results = retriever.retrieve(
      queries=queries, rerank_query=job_description, top_n=top_n, **retrieve_kwargs
    )
    ranked_ids = [r.candidate_id for r in results]

    for k in k_values:
      recalls[k].append(recall_at_k(ranked_ids, ground_truth_id, k))
    mrrs.append(mrr(ranked_ids, ground_truth_id))

  report = {f"recall@{k}": sum(v) / len(v) for k, v in recalls.items()}
  report["mrr"] = sum(mrrs) / len(mrrs)
  report["n_queries"] = len(test_set)
  return report


CROSS_ENCODER_MODELS = {
  "bge": RERANKER_DEFAULT_MODEL,
  "minilm": RERANKER_ALTERNATIVE_MODEL,
}

METHODS = {
  "semantic": {"retrieval_mode": "semantic", "use_rerank": False},
  "keyword": {"retrieval_mode": "keyword", "use_rerank": False},
  "hybrid": {"retrieval_mode": "hybrid", "use_rerank": False},
  "hybrid_rerank_bge": {"retrieval_mode": "hybrid", "use_rerank": True, "cross_encoder_key": "bge"},
  "hybrid_rerank_minilm": {"retrieval_mode": "hybrid", "use_rerank": True, "cross_encoder_key": "minilm"},
}

QUERY_MODES = {"jd_full": "job_description", "facets": "facets"}


def run_evaluation_grid(
  retrievers: dict,
  test_sets: dict,
  k_values: list = (1, 3, 5, 10),
) -> pd.DataFrame:
  """
  Evaluates every (pool, query_mode, method) combination and returns one tidy row
  per combination: pool | query_mode | method | recall@1 | recall@3 | ... | mrr | n_queries.

  retrievers: {pool_name: HybridCandidateRetriever}, one instance per pool (built
  once — cross-encoder swapping happens per call, not by rebuilding the retriever).
  test_sets: {pool_name: list from load_test_set(...)}, ideally already carrying a
  "facets" field (via add_facets_to_synthetic_jds) if query_mode="facets" will run.
  """
  from src.search.reranker import load_cross_encoder

  cross_encoders = {key: load_cross_encoder(model_name) for key, model_name in CROSS_ENCODER_MODELS.items()}

  rows = []
  for pool_name, retriever in retrievers.items():
    test_set = test_sets[pool_name]
    for query_mode, query_field in QUERY_MODES.items():
      for method_name, cfg in METHODS.items():
        cross_encoder = cross_encoders.get(cfg.get("cross_encoder_key"))
        report = evaluate_retrieval(
          test_set,
          retriever,
          k_values=k_values,
          query_field=query_field,
          retrieval_mode=cfg["retrieval_mode"],
          use_rerank=cfg["use_rerank"],
          cross_encoder=cross_encoder,
        )
        rows.append({"pool": pool_name, "query_mode": query_mode, "method": method_name, **report})

  return pd.DataFrame(rows)


class JobDescriptionFacets(BaseModel):
  facets: list[str] = Field(
    ...,
    description="Between 1 and n bullets, each a self-contained technical requirement/responsibility.",
  )


PROMPT_GENERATE_FACETS = """You are an expert in job description analysis, preparing search queries for a resume-matching system.

Your task: read the raw job description below and produce exactly {n} bullets that summarize ONLY its technical requirements, responsibilities, and projects — nothing else.

Each bullet will be used as an independent search query, so:
- Each bullet must be self-contained and cover ONE coherent technical theme (e.g. "cloud platform + pipelines", or "databases + data warehousing"). Never combine two unrelated themes into the same bullet just to hit the target count (e.g. never merge a technical tool/skill with a soft skill, a language requirement, or an unrelated technical domain).
- If the job description has more than {n} distinct requirements, group only the ones that are naturally related (same technology stack, same type of task) into a single bullet — do not force-merge unrelated ones.
- If the job description has fewer than {n} distinct requirements, return fewer bullets rather than splitting one requirement artificially or inventing content.

Include:
- Tasks and duties the candidate will perform.
- Activities, functions, and areas of responsibility associated with the role.
- Projects, objectives, or business applications the candidate will work on.
- Required or preferred professional experience (including years of experience, if stated).
- Required technical knowledge, tools, platforms, or technologies.

Exclude entirely — do not extract, and never fold into another bullet:
- Company description, mission, values, or culture.
- Statements about career development, growth opportunities, or "what we offer".
- Benefits, salary, compensation, insurance, holidays, or working conditions.
- Generic promotional or recruiting language.
- Required proficiency in spoken/written languages (e.g. "English B2").
- Soft skills mentioned in isolation (e.g. "communication skills", "teamwork") — only include them if they are inseparable from a technical responsibility in the original text.

Rules:
- Do not invent, infer, or add requirements not explicitly present in the text.
- Preserve the original technical wording as closely as possible instead of paraphrasing.
- Return ONLY the bullets, nothing else.

Job description:
{job_description}
"""


def generate_facets(job_description: str, n: int = 5, provider: str = None) -> list:
  """Splits a job description into `n` self-contained technical facets via the LLM."""
  result = chat_structured(
    system_prompt=PROMPT_GENERATE_FACETS.format(n=n, job_description=job_description),
    user_message=job_description,
    schema=JobDescriptionFacets,
    provider=provider,
  )
  return result.facets


def add_facets_to_synthetic_jds(
  jds_dir: str | Path,
  n: int = 5,
  provider: str = None,
  skip_existing: bool = True,
) -> None:
  """
  Reads every synthetic-JD JSON in jds_dir, generates facets for its
  job_description with generate_facets(), and writes them back into the same
  file under the "facets" key — so every evaluation run reuses the same facets
  instead of regenerating (and re-paying for) them each time.

  skip_existing: if True (default), files that already have a "facets" key are
  left untouched.
  """
  jds_dir = Path(jds_dir)

  for path in sorted(jds_dir.glob("*.json")):
    payload = json.loads(path.read_text(encoding="utf-8"))

    if skip_existing and "facets" in payload:
      print(f"[skip] {path.name} (ya tiene facets)")
      continue

    facets = generate_facets(payload["job_description"], n=n, provider=provider)
    payload["facets"] = facets
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] {path.name} -> {len(facets)} facets")

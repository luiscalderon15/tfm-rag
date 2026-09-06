from dataclasses import dataclass, field

import numpy as np

from src.search.fussion import reciprocal_rank_fusion, RRF_K
from src.search.keywords import build_bm25_index, CHUNK_ID_FIELD
from src.search.reranker import cross_encoder as default_cross_encoder
from src.rollup import rollup_chunks_to_candidates, fuse_candidate_facets, CANDIDATE_ID_FIELD

SEMANTIC_K = 30
KEYWORD_K = 30
SHORTLIST_SIZE = 15
MAX_EVIDENCE_PER_CANDIDATE = 3
RETRIEVAL_MODES = ("semantic", "keyword", "hybrid")


@dataclass
class Evidence:
  facet: str
  chunk_id: str
  chunk_text: str
  fusion_score: float
  rerank_score: float = None


@dataclass
class CandidateResult:
  candidate_id: str
  score: float
  evidence: list = field(default_factory=list)
  years_exp: int = None
  skills: list = field(default_factory=list)
  certifications: list = field(default_factory=list)
  about_me: str = None


class HybridCandidateRetriever:
  """
  Two-stage hybrid retriever:
    1. Chunk-level recall per query/facet: BM25 + FAISS fused via RRF.
    2. Candidate-level ranking: MaxSim rollup per facet, RRF fusion across facets,
       then cross-encoder rerank of the shortlist.
  """

  def __init__(
    self,
    vectorstore,
    candidate_id_field=CANDIDATE_ID_FIELD,
    chunk_id_field=CHUNK_ID_FIELD,
    semantic_k=SEMANTIC_K,
    keyword_k=KEYWORD_K,
    rrf_k=RRF_K,
    shortlist_size=SHORTLIST_SIZE,
    max_evidence_per_candidate=MAX_EVIDENCE_PER_CANDIDATE,
    cross_encoder=None,
    use_rerank=False,
  ):
    self.vectorstore = vectorstore
    self.candidate_id_field = candidate_id_field
    self.chunk_id_field = chunk_id_field
    self.semantic_k = semantic_k
    self.keyword_k = keyword_k
    self.rrf_k = rrf_k
    self.shortlist_size = shortlist_size
    self.max_evidence_per_candidate = max_evidence_per_candidate
    self.use_rerank = use_rerank
    self.cross_encoder = cross_encoder or default_cross_encoder

    self.docs_by_chunk_id, self.bm25, self.chunk_ids = build_bm25_index(vectorstore, chunk_id_field)
    self.years_exp_by_candidate = {
      doc.metadata[candidate_id_field]: doc.metadata.get("years_exp")
      for doc in self.docs_by_chunk_id.values()
    }
    self.skills_by_candidate = {
      doc.metadata[candidate_id_field]: doc.metadata.get("skills") or []
      for doc in self.docs_by_chunk_id.values()
    }
    self.certifications_by_candidate = {
      doc.metadata[candidate_id_field]: doc.metadata.get("certifications") or []
      for doc in self.docs_by_chunk_id.values()
    }
    self.about_me_by_candidate = {
      doc.metadata[candidate_id_field]: doc.metadata.get("about_me")
      for doc in self.docs_by_chunk_id.values()
    }
    self.full_cv_by_candidate = {
      doc.metadata[candidate_id_field]: doc.metadata.get("anonimized_raw_resume")
      for doc in self.docs_by_chunk_id.values()
    }

  def get_candidate(self, candidate_id: str) -> CandidateResult:
    """
    Fetches one candidate directly by ID, bypassing retrieval entirely — for a
    chatbot that already knows which candidate it wants to ask about. Unlike
    retrieve(), there is no query to rank fragments against, so the evidence is
    the candidate's full anonymized CV as a single entry, not retrieved fragments.
    """
    if candidate_id not in self.full_cv_by_candidate:
      raise ValueError(f"No candidate found with candidate_id={candidate_id!r}.")

    full_text = self.full_cv_by_candidate[candidate_id]
    evidence = [
      Evidence(facet="full_cv", chunk_id=f"{candidate_id}-full", chunk_text=full_text, fusion_score=None)
    ]

    return CandidateResult(
      candidate_id=candidate_id,
      score=None,
      evidence=evidence,
      years_exp=self.years_exp_by_candidate.get(candidate_id),
      skills=self.skills_by_candidate.get(candidate_id, []),
      certifications=self.certifications_by_candidate.get(candidate_id, []),
      about_me=self.about_me_by_candidate.get(candidate_id),
    )

  def _normalize_queries(self, queries):
    if isinstance(queries, str):
      return [("global", queries)]
    facets = []
    for i, query in enumerate(queries):
      if isinstance(query, (tuple, list)):
        facets.append((query[0], query[1]))
      else:
        facets.append((f"facet_{i + 1}", query))
    return facets

  def _hybrid_chunk_search(self, query, retrieval_mode="hybrid"):
    if retrieval_mode not in RETRIEVAL_MODES:
      raise ValueError(f"Unknown retrieval_mode: {retrieval_mode!r}. Expected one of {RETRIEVAL_MODES}.")

    ranked_lists = []

    if retrieval_mode in ("semantic", "hybrid"):
      semantic_hits = self.vectorstore.similarity_search_with_score(query, k=self.semantic_k)
      ranked_lists.append([doc.metadata[self.chunk_id_field] for doc, _ in semantic_hits])

    if retrieval_mode in ("keyword", "hybrid"):
      tokenized_query = query.lower().split()
      bm25_scores = self.bm25.get_scores(tokenized_query)
      top_idx = np.argsort(bm25_scores)[::-1][: self.keyword_k]
      ranked_lists.append([self.chunk_ids[i] for i in top_idx])

    return reciprocal_rank_fusion(ranked_lists, k=self.rrf_k)

  def retrieve(self, queries, rerank_query=None, top_n=10, use_rerank=None, retrieval_mode="hybrid", cross_encoder=None):
    """
    queries: a single query string, or a list of queries/facets to fuse. Each list
    item is either a plain string (auto-named "facet_1", "facet_2", ...) or an
    explicit (facet_name, query_text) tuple. Sub-query generation from a job
    description is the caller's responsibility, not this module's.

    use_rerank: overrides the instance default (self.use_rerank) for this call only,
    e.g. to let a UI toggle reranking on/off without rebuilding the retriever.

    retrieval_mode: "hybrid" (default, semantic+keyword fused via RRF), "semantic",
    or "keyword" — isolates one signal for ablation studies.

    cross_encoder: overrides the instance default (self.cross_encoder) for this call
    only, e.g. to A/B test two reranker models without rebuilding the retriever.
    """
    use_rerank = self.use_rerank if use_rerank is None else use_rerank
    facets = self._normalize_queries(queries)
    facet_names = [name for name, _ in facets]

    per_facet_candidate_best = []
    for _, facet_query in facets:
      fused_chunk_scores = self._hybrid_chunk_search(facet_query, retrieval_mode=retrieval_mode)
      candidate_best = rollup_chunks_to_candidates(
        fused_chunk_scores, self.docs_by_chunk_id, self.candidate_id_field
      )
      per_facet_candidate_best.append(candidate_best)

    fused_candidate_scores = fuse_candidate_facets(per_facet_candidate_best, k=self.rrf_k)
    shortlist_ids = sorted(fused_candidate_scores, key=fused_candidate_scores.get, reverse=True)[: self.shortlist_size]

    max_evidence = max(self.max_evidence_per_candidate, len(facet_names))
    results = self._build_candidate_results(shortlist_ids, fused_candidate_scores, per_facet_candidate_best, facet_names, max_evidence)

    if use_rerank:
      rerank_text = rerank_query if rerank_query is not None else facets[0][1]
      results = self._rerank(rerank_text, results, cross_encoder=cross_encoder)

    results.sort(key=lambda r: r.score, reverse=True)
    return results[:top_n]

  def _build_candidate_results(self, shortlist_ids, fused_scores, per_facet_candidate_best, facet_names, max_evidence):
    results = []
    for candidate_id in shortlist_ids:
      evidence_by_chunk = {}
      for facet_name, candidate_best in zip(facet_names, per_facet_candidate_best):
        if candidate_id in candidate_best:
          score, chunk_id = candidate_best[candidate_id]
          if chunk_id in evidence_by_chunk:
            # Same chunk already won for another facet — merge instead of duplicating.
            existing = evidence_by_chunk[chunk_id]
            existing.facet = f"{existing.facet}, {facet_name}"
            existing.fusion_score = max(existing.fusion_score, score)
          else:
            doc = self.docs_by_chunk_id[chunk_id]
            evidence_by_chunk[chunk_id] = Evidence(facet=facet_name, chunk_id=chunk_id, chunk_text=doc.page_content, fusion_score=score)
      evidence = list(evidence_by_chunk.values())
      evidence.sort(key=lambda e: e.fusion_score, reverse=True)
      evidence = evidence[:max_evidence]
      years_exp = self.years_exp_by_candidate.get(candidate_id)
      skills = self.skills_by_candidate.get(candidate_id, [])
      certifications = self.certifications_by_candidate.get(candidate_id, [])
      about_me = self.about_me_by_candidate.get(candidate_id)
      results.append(CandidateResult(
        candidate_id=candidate_id,
        score=fused_scores[candidate_id],
        evidence=evidence,
        years_exp=years_exp,
        skills=skills,
        certifications=certifications,
        about_me=about_me,
      ))
    return results

  def _rerank(self, rerank_query, results, cross_encoder=None):
    cross_encoder = cross_encoder or self.cross_encoder

    pairs = []
    pair_owners = []
    for result in results:
      for evidence in result.evidence:
        pairs.append((rerank_query, evidence.chunk_text))
        pair_owners.append(evidence)

    if not pairs:
      return results

    rerank_scores = cross_encoder.predict(pairs)
    for evidence, score in zip(pair_owners, rerank_scores):
      evidence.rerank_score = float(score)

    for result in results:
      scored_evidence = [e.rerank_score for e in result.evidence if e.rerank_score is not None]
      if scored_evidence:
        result.score = max(scored_evidence)

    return results

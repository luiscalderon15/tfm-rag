from config import CANDIDATE_ID_FIELD, RRF_K
from src.search.fussion import reciprocal_rank_fusion

def rollup_chunks_to_candidates(fused_chunk_scores, docs_by_chunk_id, candidate_id_field=CANDIDATE_ID_FIELD):
  """MaxSim rollup: keep each candidate's single best-scoring chunk for one query/facet."""
  best = {}
  for chunk_id, score in fused_chunk_scores.items():
    doc = docs_by_chunk_id.get(chunk_id)
    if doc is None:
      continue
    candidate_id = doc.metadata.get(candidate_id_field)
    if candidate_id is None:
      continue
    if candidate_id not in best or score > best[candidate_id][0]:
      best[candidate_id] = (score, chunk_id)
  return best

def fuse_candidate_facets(per_facet_candidate_best, k=RRF_K):
  """RRF across facets: candidates covering more distinct JD facets rank higher than a single perfect match."""
  ranked_lists = []
  for candidate_best in per_facet_candidate_best:
    ranked = sorted(candidate_best, key=lambda cid: candidate_best[cid][0], reverse=True)
    ranked_lists.append(ranked)
  return reciprocal_rank_fusion(ranked_lists, k=k)
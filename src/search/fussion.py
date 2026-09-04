RRF_K = 60
def reciprocal_rank_fusion(ranked_id_lists:list[list], k:int=RRF_K):
  """Fuse multiple best-first ranked id lists into a single score dict via RRF."""
  scores = {}
  for ranked_ids in ranked_id_lists:
    for rank, item_id in enumerate(ranked_ids):
      scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank + 1)
  return scores
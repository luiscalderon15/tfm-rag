
#CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"
CROSS_ENCODER_MODEL = "BAAI/bge-reranker-v2-m3"
_cross_encoders = {}

def load_cross_encoder(model_name=CROSS_ENCODER_MODEL):
  if model_name not in _cross_encoders:
    from sentence_transformers import CrossEncoder
    _cross_encoders[model_name] = CrossEncoder(model_name, device="mps")
  return _cross_encoders[model_name]

cross_encoder = load_cross_encoder()
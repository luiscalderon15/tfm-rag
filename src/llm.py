import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from config import MODELS_CONFIG, LLM_PROVIDER

load_dotenv()

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")

_clients = {}


def _get_model_config(name: str) -> dict:
  cfg = MODELS_CONFIG.get(name)
  if not cfg or "deployment_name" not in cfg:
    raise ValueError(
      f"Unknown model config: {name!r}. Expected one of the llm_* blocks in "
      f"config/models.yml (e.g. 'llm_opensource', 'llm_nano', 'llm_gpt')."
    )
  return cfg


def _build_client(name: str, temperature):
  cfg = _get_model_config(name)
  backend = cfg["provider"]
  deployment_name = cfg["deployment_name"]

  if backend == "ollama":
    # No num_predict cap — let the model use its own default (qwen2.5:7b-instruct
    # has a 32K context window). A 1024-token cap here previously truncated
    # evaluations mid-response once top_n grew past ~10 candidates.
    return ChatOllama(model=deployment_name, temperature=temperature if temperature is not None else 0)

  if backend == "azure":
    if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY:
      raise RuntimeError(
        f"Model config {name!r} requires AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY "
        "to be set (e.g. in a local .env file)."
      )
    kwargs = dict(model=deployment_name, base_url=AZURE_OPENAI_ENDPOINT, api_key=AZURE_OPENAI_API_KEY)
    if temperature is not None:
      kwargs["temperature"] = temperature
    return ChatOpenAI(**kwargs)

  raise ValueError(f"Unknown provider {backend!r} for model config {name!r}.")


def _get_client(name: str, temperature):
  cache_key = (name, temperature)
  if cache_key not in _clients:
    _clients[cache_key] = _build_client(name, temperature)
  return _clients[cache_key]


def chat_structured(
  system_prompt: str,
  user_message: str,
  schema: type[BaseModel],
  provider: str = None,
  temperature: float = None,
) -> BaseModel:
  """
  Calls the model configured under `provider` — a key in config/models.yml's
  llm_* blocks (e.g. "llm_opensource", "llm_nano", "llm_gpt") — and returns a
  validated instance of `schema`.

  provider: overrides LLM_PROVIDER for this call only.
  temperature: overrides that model's configured temperature for this call
  only. Ignored if the model's config has `temperature: null` (e.g. the
  o4-mini reasoning model, which rejects a custom temperature outright).
  """
  model_name = provider or LLM_PROVIDER
  cfg = _get_model_config(model_name)
  backend = cfg["provider"]
  supports_temperature = cfg.get("temperature") is not None

  effective_temperature = None
  if supports_temperature:
    effective_temperature = temperature if temperature is not None else cfg.get("temperature")

  client = _get_client(model_name, effective_temperature)

  if backend == "ollama":
    # method="json_schema" pins Ollama's native grammar-constrained JSON decoding
    # (always syntactically valid) instead of LangChain's function-calling-based
    # structured output, which we verified can silently return None on this model.
    structured_client = client.with_structured_output(schema, method="json_schema")
  else:
    structured_client = client.with_structured_output(schema)

  return structured_client.invoke([
    SystemMessage(content=system_prompt),
    HumanMessage(content=user_message),
  ])

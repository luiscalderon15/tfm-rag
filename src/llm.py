import os

import ollama
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")

MODEL_OPENSOURCE = "qwen2.5"
MODEL_OPENSOURCE_CHAT = "qwen2.5:7b-instruct"

AZURE_OPENAI_MODEL = os.getenv("AZURE_OPENAI_MODEL", "o4-mini")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")

_azure_client = None


def _get_azure_client() -> ChatOpenAI:
  global _azure_client
  if _azure_client is None:
    if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY:
      raise RuntimeError(
        "LLM_PROVIDER=azure requires AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY "
        "to be set (e.g. in a local .env file, see .env.example)."
      )
    _azure_client = ChatOpenAI(
      model=AZURE_OPENAI_MODEL,
      base_url=AZURE_OPENAI_ENDPOINT,
      api_key=AZURE_OPENAI_API_KEY,
    )
  return _azure_client


def chat_structured(
  system_prompt: str,
  user_message: str,
  schema: type[BaseModel],
  provider: str = None,
  temperature: float = 0,
) -> BaseModel:
  """
  Calls the configured chat LLM and returns a validated instance of `schema`.

  provider: overrides LLM_PROVIDER for this call only ("ollama" or "azure").
  temperature: only honored by the "ollama" provider — Azure's o4-mini is a
  reasoning model and does not accept a custom temperature.
  """
  provider = provider or LLM_PROVIDER

  if provider == "ollama":
    response = ollama.chat(
      model=MODEL_OPENSOURCE_CHAT,
      messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
      ],
      format=schema.model_json_schema(),
      options={"temperature": temperature},
    )
    return schema.model_validate_json(response["message"]["content"])

  if provider == "azure":
    client = _get_azure_client().with_structured_output(schema)
    return client.invoke([
      SystemMessage(content=system_prompt),
      HumanMessage(content=user_message),
    ])

  raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}. Expected 'ollama' or 'azure'.")

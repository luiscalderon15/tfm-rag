from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

MODEL_OPENSOURCE = "qwen2.5"
MODEL_OPENSOURCE_CHAT = "qwen2.5:7b-instruct"

llm = ChatOllama(
    model=MODEL_OPENSOURCE,
    temperature=0
)
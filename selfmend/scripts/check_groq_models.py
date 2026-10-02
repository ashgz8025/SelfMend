"""List all Groq models accessible with the current API key, then test each chat model."""
import os, sys, requests
from dotenv import load_dotenv
load_dotenv()

key = os.getenv("GROQ_API_KEY", "")
if not key or "REPLACE" in key:
    print("GROQ_API_KEY not set"); sys.exit(1)

r = requests.get(
    "https://api.groq.com/openai/v1/models",
    headers={"Authorization": f"Bearer {key}"},
    timeout=10,
)
print(f"Status: {r.status_code}")
data = r.json()
models = sorted(m["id"] for m in data.get("data", []))
print(f"\nAll accessible models ({len(models)}):")
for m in models:
    print(f"  {m}")

# Find usable chat LLMs (exclude whisper, guard models, etc.)
chat_models = [m for m in models
               if not any(x in m for x in ("whisper", "guard", "orpheus"))]
print(f"\nChat-capable candidates: {chat_models}")

# Quick smoke-test the first viable one
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

for model_id in chat_models[:3]:
    try:
        llm = ChatGroq(model=model_id, temperature=0.0, api_key=key, max_retries=1)
        resp = llm.invoke([HumanMessage(content="Reply with just: OK")])
        print(f"\n  {model_id}: OK -> {resp.content.strip()[:40]}")
        break
    except Exception as e:
        print(f"\n  {model_id}: FAIL -> {str(e)[:80]}")

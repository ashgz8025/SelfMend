"""Test which Groq models reliably return structured JSON for the diagnoser."""
import os, sys, json, re
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

key = os.getenv("GROQ_API_KEY", "")

PROMPT = """You are a data quality expert.

FAILURE: null spike in Price Per Unit column. Null rate is 50%, baseline is 10%.

EVIDENCE:
Incident inc-004: Price Per Unit null rate spiked to 50%. Root cause: POS timeout.
Fix applied: impute_nulls. Outcome: fixed.

WHITELISTED ACTIONS: normalize_sentinels_to_null, restore_total_from_quantity_price, impute_nulls, drop_duplicates, clip_outliers

Reply with valid JSON only:
{
  "root_cause": "<one sentence>",
  "evidence_used": ["inc-004"],
  "proposed_action": "impute_nulls",
  "confidence": 0.85,
  "risk_level": "low"
}"""

models = ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b", "allam-2-7b"]

for model_id in models:
    try:
        llm = ChatGroq(model=model_id, temperature=0.0, api_key=key, max_retries=1)
        resp = llm.invoke([
            SystemMessage(content="You are a data quality expert. Reply with valid JSON only."),
            HumanMessage(content=PROMPT),
        ])
        raw = resp.content.strip()
        # Try parse
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            parsed = json.loads(m.group())
            action = parsed.get("proposed_action", "?")
            conf   = parsed.get("confidence", "?")
            risk   = parsed.get("risk_level", "?")
            print(f"  PASS  {model_id:<30}  action={action}  conf={conf}  risk={risk}")
        else:
            print(f"  NOJSON {model_id:<30}  raw={raw[:80]}")
    except Exception as e:
        print(f"  FAIL  {model_id:<30}  {str(e)[:80]}")

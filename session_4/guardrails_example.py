from guardrails import Guard, OnFailAction
from guardrails.hub import DetectPII, ToxicLanguage, ValidLength
import re

# ── 1. PII Detection and Redaction ────────────────────────────
pii_guard = Guard().use(
    DetectPII,
    pii_types=["EMAIL_ADDRESS","PHONE_NUMBER","CREDIT_CARD","NIF"],
    on_fail=OnFailAction.FIX,   # auto-redact instead of raising
)

def safe_response(user_input: str, llm_response: str) -> str:
    validated, _, _ = pii_guard.validate(llm_response)
    return validated  # PII redacted, e.g. "Call <PHONE_NUMBER>"

# ── 2. Hallucination detection (NLI-based) ───────────────────
from transformers import pipeline

nli = pipeline("text-classification", model="cross-encoder/nli-deberta-v3-small")

def faithfulness_guard(context: str, answer: str, threshold=0.5) -> bool:
    """Returns True if answer is entailed by context."""
    result = nli(f"{context} [SEP] {answer}")
    entailment_score = next(r["score"] for r in result if r["label"] == "ENTAILMENT")
    if entailment_score < threshold:
        raise ValueError(f"Hallucination detected: entailment={entailment_score:.2f}")
    return True

# ── 3. Toxicity filter (input + output) ─────────────────────
tox_guard = Guard().use(ToxicLanguage, threshold=0.5, validation_method="sentence")

# ── 4. Prompt injection detection ────────────────────────────
INJECTION_PATTERNS = [
    r"ignore (previous|above|all) instructions",
    r"you are now (DAN|a different|an evil)",
    r"repeat after me",
]
def detect_prompt_injection(user_input: str) -> bool:
    return any(re.search(p, user_input, re.IGNORECASE) for p in INJECTION_PATTERNS)

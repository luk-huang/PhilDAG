# query_bridge.py
from __future__ import annotations
from typing import List, Tuple, Dict, Any, Optional, Callable
from dataclasses import dataclass
import json, os, re, math
from dotenv import load_dotenv
load_dotenv()

# --- Your schema ---
from schemas.schema import Statement, Argument, Artifact, Quote

# Compatibility alias used throughout this module
Claim = Statement

# =============== Utilities ===============

def _cosine(u, v) -> float:
    num = sum(a*b for a, b in zip(u, v))
    du = math.sqrt(sum(a*a for a in u))
    dv = math.sqrt(sum(b*b for b in v))
    return 0.0 if du == 0 or dv == 0 else num / (du * dv)

class EmbeddingBackend:
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError

class SentenceTransformerEmbeddings(EmbeddingBackend):
    """
    Offline-friendly, pip install sentence-transformers
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # type: ignore
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, normalize_embeddings=False).tolist()

# =============== LLM plumbing ===============

@dataclass
class LLMResult:
    answer: str
    highlight_claim_ids: List[int]
    highlight_argument_ids: List[int]
    derived_claim_ids: List[int]  # optional, possibly empty
    notes: str                    # optional, possibly empty

def _default_system_prompt() -> str:
    return (
        "You are a precise reasoning assistant working over a *directed acyclic graph (DAG)* of beliefs.\n"
        "The graph consists of:\n"
        "- Claims (nodes), each with a unique integer id and a short description.\n"
        "- Arguments (hyperedges), each with a unique integer id, a list of premise claim ids, "
        "  and a single conclusion claim id. Semantics: if *all* premises are true, the conclusion is supported.\n\n"
        "Your job: given a natural-language query and the graph (subset), produce:\n"
        "1) A concise direct answer to the query (<= 6 sentences, neutral tone).\n"
        "2) The ids of claims to *highlight* as most relevant evidence (they might be premises, conclusions, or axioms).\n"
        "3) The ids of arguments to highlight (those that most directly connect the highlighted claims to the answer).\n"
        "4) Optionally, ids of claims that are *derivable* in light of this query (if the query implies some forward chaining).\n"
        "5) A brief note field (1-3 sentences) with justification using claim ids only. DO NOT reveal chain-of-thought.\n\n"
        "Rules:\n"
        "- ONLY use claims/arguments that appear in the provided graph chunk. Do not invent ids or content.\n"
        "- Prefer minimal, non-redundant highlight sets that make the reasoning legible.\n"
        "- If the query asks for contradictions, choose pairs of claims that conflict (semantic opposition) and highlight the linking arguments if relevant.\n"
        "- If nothing in the graph answers the question, say so and return empty highlight lists.\n"
        "- Output STRICT JSON conforming to the schema provided, with keys in lower_snake_case.\n"
    )

def _response_schema_text() -> str:
    # A self-describing JSON schema the model can follow.
    return """
Return JSON exactly like this (no prose before or after):

{
  "answer": "string, <= 6 sentences, neutral and direct.",
  "highlight_claim_ids": [1, 2, 3],
  "highlight_argument_ids": [10, 11],
  "derived_claim_ids": [4, 5],
  "notes": "brief justification using claim ids only, e.g. 'Used C1,C2 -> A10 -> C5'."
}

Where:
- highlight_claim_ids: integer ids of claims to light up
- highlight_argument_ids: integer ids of arguments to light up
- derived_claim_ids: integer ids of conclusions that follow given the query context (can be empty)
- If nothing applies, use empty arrays and a helpful 'answer'.
"""

def _serialize_graph_for_llm(claims: List[Claim], arguments: List[Argument]) -> Dict[str, Any]:
    # Compact, ID-first JSON the LLM can reason over reliably.
    def _artifact_payload(artifacts: List[Artifact]):
        if not artifacts:
            return None
        art = artifacts[0]
        return {
            "id": art.id,
            "name": art.name,
            "author": art.author,
            "title": getattr(art, "title", getattr(art, "tile", "")),
            "year": art.year,
        }

    claims_json = [
        {
            "id": c.id,
            "desc": getattr(c, "desc", c.statement),
            "artifact": _artifact_payload(getattr(c, "artifact", [])),
        }
        for c in claims
    ]
    args_json = [
        {
            "id": a.id,
            "premise_ids": [p.id for p in a.premise],
            "conclusion_id": a.conclusion.id,
            "desc": a.desc,
        }
        for a in arguments
    ]
    return {"claims": claims_json, "arguments": args_json}

def _build_prompt(query: str, claims: List[Claim], arguments: List[Argument]) -> List[Dict[str, str]]:
    sys = _default_system_prompt()
    schema = _response_schema_text()
    graph_json = _serialize_graph_for_llm(claims, arguments)
    user = {
        "role": "user",
        "content": (
            f"QUERY:\n{query}\n\n"
            f"GRAPH:\n{json.dumps(graph_json, ensure_ascii=False)}\n\n"
            f"{schema}"
        )
    }
    return [{"role": "system", "content": sys}, user]

def _extract_first_json(s: str) -> str:
    """
    Robustly extract the first {...} JSON object from a model response.
    """
    # Fast path: already a JSON object
    s_strip = s.strip()
    if s_strip.startswith("{") and s_strip.endswith("}"):
        return s_strip
    # Fallback: regex the first balanced-looking JSON object
    match = re.search(r"\{(?:[^{}]|(?R))*\}", s, flags=re.DOTALL)
    if match:
        return match.group(0)
    # Last resort: try to trim before first '{' and after last '}'
    if "{" in s and "}" in s:
        start, end = s.find("{"), s.rfind("}")
        return s[start:end+1]
    raise ValueError("Could not locate JSON in model output.")

def _parse_llm_json(payload: str) -> LLMResult:
    obj = json.loads(payload)
    # Soft validation with defaults
    answer = obj.get("answer", "").strip()
    h_claims = [int(x) for x in obj.get("highlight_claim_ids", []) if isinstance(x, (int, str))]
    h_args = [int(x) for x in obj.get("highlight_argument_ids", []) if isinstance(x, (int, str))]
    d_claims = [int(x) for x in obj.get("derived_claim_ids", []) if isinstance(x, (int, str))]
    notes = obj.get("notes", "").strip()
    return LLMResult(answer=answer, highlight_claim_ids=h_claims,
                     highlight_argument_ids=h_args, derived_claim_ids=d_claims, notes=notes)

# =============== Optional prefiltering ===============
def _prefilter_graph_by_embeddings(
    query: str,
    claims: List[Claim],
    arguments: List[Argument],
    embedder: Optional[EmbeddingBackend],
    max_claims: int = 200,
    max_arguments: int = 400,
) -> Tuple[List[Claim], List[Argument]]:
    """
    Keeps payloads small. If embedder is provided, rank claims by similarity to query.
    Then keep any argument whose premises or conclusion survive.
    """
    if embedder is None or len(claims) <= max_claims:
        return claims, arguments

    texts = [getattr(c, "desc", c.statement) for c in claims] + [query]
    vecs = embedder.embed_texts(texts)
    qv = vecs[-1]
    scores = [(_cosine(v, qv), i) for i, v in enumerate(vecs[:-1])]
    scores.sort(reverse=True)
    keep_idx = {i for _, i in scores[:max_claims]}
    kept_claims = [c for i, c in enumerate(claims) if i in keep_idx]
    kept_ids = {c.id for c in kept_claims}

    kept_args = [
        a for a in arguments
        if a.conclusion.id in kept_ids or any(p.id in kept_ids for p in a.premise)
    ]
    if len(kept_args) > max_arguments:
        kept_args = kept_args[:max_arguments]
    return kept_claims, kept_args

# =============== Public API ===============

def query_graph(
    query: str,
    claims: List[Claim],
    arguments: List[Argument],
    *,
    llm_call: Optional[Callable[[List[Dict[str, str]]], str]] = None,
    embedder: Optional[EmbeddingBackend] = None,
    prefilter: bool = True,
) -> Tuple[List[Claim], List[Argument], str]:
    """
    Calls an LLM with (query + serialized graph), expects STRICT JSON response with:
      { answer, highlight_claim_ids, highlight_argument_ids, derived_claim_ids, notes }

    Returns: (highlighted_claims, highlighted_arguments, answer_string)

    Parameters:
      - llm_call: a callable that accepts OpenAI/Anthropic-style messages and returns the raw model text.
                  If None, tries OpenAI via environment (OPENAI_API_KEY).
      - embedder: optional embedding backend to preselect top-k claims for context (keeps prompts short).
      - prefilter: disable to send the whole graph (for small graphs).
    """
    # Optional payload reduction
    c_list, a_list = (claims, arguments)
    if prefilter:
        c_list, a_list = _prefilter_graph_by_embeddings(query, claims, arguments, embedder)

    messages = _build_prompt(query, c_list, a_list)

    # --- LLM call strategy ---
    if llm_call is None:
        # Try OpenAI new SDK (client = OpenAI(); client.chat.completions.create)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("No llm_call provided and OPENAI_API_KEY not set.")
        try:
            from openai import OpenAI  # type: ignore
            client = OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
                messages=messages,
                temperature=0.2,
                max_tokens=800,
            )
            raw = resp.choices[0].message.content or ""
        except Exception as e:
            raise RuntimeError(f"OpenAI call failed: {e}")
    else:
        raw = llm_call(messages)

    # --- JSON extraction + parse with fallbacks ---
    try:
        json_text = _extract_first_json(raw)
        parsed = _parse_llm_json(json_text)
    except Exception:
        # Gentle repair attempt: tell the model to fix to schema (self-repair loop)
        repair_msgs = messages + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": "The previous output was not valid JSON. "
                                        "Please return only a single JSON object matching the required schema."}
        ]
        if llm_call is None:
            from openai import OpenAI  # type: ignore
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            retry = client.chat.completions.create(
                model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
                messages=repair_msgs,
                temperature=0.0,
                max_tokens=800,
            )
            raw_retry = retry.choices[0].message.content or ""
        else:
            raw_retry = llm_call(repair_msgs)
        json_text = _extract_first_json(raw_retry)
        parsed = _parse_llm_json(json_text)

    # Map ids -> objects (restrict to original full lists so lights align with UI)
    claim_by_id = {c.id: c for c in claims}
    arg_by_id = {a.id: a for a in arguments}
    highlight_claims = [claim_by_id[i] for i in parsed.highlight_claim_ids if i in claim_by_id]
    highlight_arguments = [arg_by_id[i] for i in parsed.highlight_argument_ids if i in arg_by_id]

    # You may also compute auto-highlights for derived ids if not returned:
    for did in parsed.derived_claim_ids:
        if did in claim_by_id and all(c.id != did for c in highlight_claims):
            highlight_claims.append(claim_by_id[did])

    return highlight_claims, highlight_arguments, parsed.answer

# ================= Examples =================

def openai_llm_call_factory(model: str = "gpt-4o-mini", temperature: float = 0.2) -> Callable[[List[Dict[str, str]]], str]:
    """
    Example adapter for OpenAI's Chat Completions API.
    Usage:
        llm = openai_llm_call_factory("gpt-4o-mini")
        query_graph("Do I endorse compatibilism?", claims, arguments, llm_call=llm)
    """
    from openai import OpenAI  # type: ignore
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    def _call(messages: List[Dict[str, str]]) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=800,
        )
        return resp.choices[0].message.content or ""
    return _call


if __name__ == "__main__":
    print("general_user_queries is intended to be imported, not run directly.")

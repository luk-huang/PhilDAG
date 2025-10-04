# query_bridge.py
from __future__ import annotations
from typing import List, Tuple, Dict, Any, Optional, Callable
from dataclasses import dataclass
import json, os, math
from collections import defaultdict
from openai import OpenAI  # type: ignore
from sentence_transformers import SentenceTransformer  # type: ignore
import json
from pathlib import Path
import argparse
from dotenv import load_dotenv

load_dotenv()

from schema import Statement, Argument, Artifact, Quote

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
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, normalize_embeddings=False).tolist()

# =============== LLM plumbing ===============

@dataclass
class LLMResult:
    answer: str
    highlight_statement_ids: List[int]
    highlight_argument_ids: List[int]
    derived_statement_ids: List[int]  # optional, possibly empty
    notes: str                        # optional, possibly empty

def _default_system_prompt() -> str:
    return (
        "You are a precise reasoning assistant working over a *directed acyclic graph (DAG)* of beliefs.\n"
        "The graph consists of:\n"
        "- Statements (nodes), each with a unique integer id and short text.\n"
        "- Arguments (hyperedges), each with a unique integer id, a list of premise statement ids,\n"
        "  and a single conclusion statement id. Semantics: if *all* premises are true, the conclusion is supported.\n\n"
        "Your job: given a natural-language query and the graph (subset), produce:\n"
        "1) A concise direct answer to the query (<= 6 sentences, neutral tone).\n"
        "2) The ids of statements to *highlight* as most relevant evidence (premises, conclusions, or axioms).\n"
        "3) The ids of arguments to highlight (those that most directly connect the highlighted statements to the answer).\n"
        "4) Optionally, ids of statements that are *derivable* in light of this query (if the query implies forward chaining).\n"
        "5) A brief note field (1-3 sentences) with justification using statement/argument ids only. DO NOT reveal chain-of-thought.\n\n"
        "Rules:\n"
        "- ONLY use statements/arguments that appear in the provided graph chunk. Do not invent ids or content.\n"
        "- Prefer minimal, non-redundant highlight sets that make the reasoning legible.\n"
        "- If the query asks for contradictions, choose pairs of statements that conflict and highlight the linking arguments if relevant.\n"
        "- If nothing in the graph answers the question, say so and return empty highlight lists.\n"
        "- Output STRICT JSON conforming to the schema provided, with keys in lower_snake_case.\n"
    )

def _response_schema_text() -> str:
    # A self-describing JSON schema the model can follow.
    return """
Return JSON exactly like this (no prose before or after):

{
  "answer": "string, <= 6 sentences, neutral and direct.",
  "highlight_statement_ids": [1, 2, 3],
  "highlight_argument_ids": [10, 11],
  "derived_statement_ids": [4, 5],
  "notes": "brief justification using ids only, e.g. 'Used S1,S2 -> A10 -> S5'."
}

Where:
- highlight_statement_ids: integer ids of statements to light up
- highlight_argument_ids: integer ids of arguments to light up
- derived_statement_ids: integer ids of conclusions that follow given the query context (can be empty)
- If nothing applies, use empty arrays and a helpful 'answer'.
"""

def _serialize_graph_for_llm(statements: List[Statement], arguments: List[Argument]) -> Dict[str, Any]:
    # Compact, ID-first JSON the LLM can reason over reliably.
    stmts_json = []
    for s in statements:
        # artifacts is a list[Artifact]
        arts = []
        if getattr(s, "artifact", None):
            for a in s.artifact:
                arts.append({
                    "id": a.id,
                    "name": a.name,
                    "author": a.author,
                    "title": a.title,
                    "year": a.year,
                })
        cites = []
        if getattr(s, "citations", None):
            for q in s.citations:
                cites.append({"page": q.page, "text": q.text})

        stmts_json.append({
            "id": s.id,
            "text": s.statement,
            "artifacts": arts,       # list, may be empty
            "citations": cites,      # list, may be empty
        })

    args_json = [
        {
            "id": a.id,
            "premise_ids": [p.id for p in a.premise],
            "conclusion_id": a.conclusion.id,
            "desc": a.desc,
        }
        for a in arguments
    ]
    return {"statements": stmts_json, "arguments": args_json}

def _build_prompt(query: str, statements: List[Statement], arguments: List[Argument]) -> List[Dict[str, str]]:
    sys = _default_system_prompt()
    schema = _response_schema_text()
    graph_json = _serialize_graph_for_llm(statements, arguments)
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
    Uses brace counting (Python's 're' lacks recursive patterns).
    """
    s = s.strip()
    # Fast path: exactly one object
    if s.startswith("{") and s.endswith("}"):
        # Quick validation
        try:
            json.loads(s)
            return s
        except Exception:
            pass

    # Scan for first balanced object
    depth = 0
    start = -1
    for i, ch in enumerate(s):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start != -1:
                    candidate = s[start:i+1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except Exception:
                        # keep scanning for next balanced candidate
                        start = -1
                        continue
    raise ValueError("Could not locate JSON in model output.")

def _parse_llm_json(payload: str) -> LLMResult:
    obj = json.loads(payload)
    # Soft validation with defaults
    answer = obj.get("answer", "").strip()
    h_statements = [int(x) for x in obj.get("highlight_statement_ids", []) if isinstance(x, (int, str))]
    h_args = [int(x) for x in obj.get("highlight_argument_ids", []) if isinstance(x, (int, str))]
    d_statements = [int(x) for x in obj.get("derived_statement_ids", []) if isinstance(x, (int, str))]
    notes = obj.get("notes", "").strip()
    return LLMResult(
        answer=answer,
        highlight_statement_ids=h_statements,
        highlight_argument_ids=h_args,
        derived_statement_ids=d_statements,
        notes=notes
    )

# =============== Optional prefiltering ===============
def _prefilter_graph_by_embeddings(
    query: str,
    statements: List[Statement],
    arguments: List[Argument],
    embedder: Optional[EmbeddingBackend],
    max_statements: int = 200,
    max_arguments: int = 400,
) -> Tuple[List[Statement], List[Argument]]:
    """
    Keeps payloads small. If embedder is provided, rank statements by similarity to query.
    Then keep any argument whose premises or conclusion survive.
    """
    if embedder is None or len(statements) <= max_statements:
        return statements, arguments

    texts = [s.statement for s in statements] + [query]
    vecs = embedder.embed_texts(texts)
    qv = vecs[-1]
    scores = [(_cosine(v, qv), i) for i, v in enumerate(vecs[:-1])]
    scores.sort(reverse=True)
    keep_idx = {i for _, i in scores[:max_statements]}
    kept_statements = [s for i, s in enumerate(statements) if i in keep_idx]
    kept_ids = {s.id for s in kept_statements}

    kept_args = [
        a for a in arguments
        if a.conclusion.id in kept_ids or any(p.id in kept_ids for p in a.premise)
    ]
    if len(kept_args) > max_arguments:
        kept_args = kept_args[:max_arguments]
    return kept_statements, kept_args

# =============== Public API ===============

def query_graph(
    query: str,
    statements: List[Statement],
    arguments: List[Argument],
    *,
    llm_call: Optional[Callable[[List[Dict[str, str]]], str]] = None,
    embedder: Optional[EmbeddingBackend] = None,
    prefilter: bool = True,
) -> Tuple[List[Statement], List[Argument], str]:
    """
    Calls an LLM with (query + serialized graph), expects STRICT JSON response with:
      { answer, highlight_statement_ids, highlight_argument_ids, derived_statement_ids, notes }

    Returns: (highlighted_statements, highlighted_arguments, answer_string)

    Parameters:
      - llm_call: a callable that accepts OpenAI/Anthropic-style messages and returns the raw model text.
                  If None, tries OpenAI via environment (OPENAI_API_KEY).
      - embedder: optional embedding backend to preselect top-k statements for context (keeps prompts short).
      - prefilter: disable to send the whole graph (for small graphs).
    """
    # Optional payload reduction
    s_list, a_list = (statements, arguments)
    if prefilter:
        s_list, a_list = _prefilter_graph_by_embeddings(query, statements, arguments, embedder)

    messages = _build_prompt(query, s_list, a_list)

    # --- LLM call strategy ---
    if llm_call is None:
        # OpenAI chat.completions API via new SDK
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("No llm_call provided and OPENAI_API_KEY not set.")
        try:
            
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
    stmt_by_id = {s.id: s for s in statements}
    arg_by_id = {a.id: a for a in arguments}
    highlight_statements = [stmt_by_id[i] for i in parsed.highlight_statement_ids if i in stmt_by_id]
    highlight_arguments = [arg_by_id[i] for i in parsed.highlight_argument_ids if i in arg_by_id]

    # Optionally add derived statements if not already highlighted:
    for did in parsed.derived_statement_ids:
        if did in stmt_by_id and all(s.id != did for s in highlight_statements):
            highlight_statements.append(stmt_by_id[did])

    return highlight_statements, highlight_arguments, parsed.answer

# ================= Examples =================

def openai_llm_call_factory(model: str = "gpt-4o-mini", temperature: float = 0.2) -> Callable[[List[Dict[str, str]]], str]:
    """
    Example adapter for OpenAI's Chat Completions API.
    Usage:
        llm = openai_llm_call_factory("gpt-4o-mini")
        query_graph("Do I endorse compatibilism?", statements, arguments, llm_call=llm)
    """
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


def process_json(path: str) -> Tuple[List[Statement], List[Argument]]:
    data = json.loads(Path(path).read_text())
    statements = []
    for s in data["statements"].values():
        s_obj = Statement(id=0, artifact=[], statement="", citations=[])
        s_obj.id = s["id"]
        s_obj.artifact = [Artifact(**art) for art in s["artifact"]]
        s_obj.citations = [Quote(**q) for q in s.get("citations", [])]
        s_obj.statement = s["statement"]
        statements.append(s_obj)
    arguments = []
    for a in data["arguments"].values():
        a_obj = Argument(id=0,premise=[],conclusion=Statement(id=0,artifact=[],statement="",citations=[]),desc="")
        for p in a["premise"]:
            p_obj = next((s for s in statements if s.id == int(p)), None)
            if p_obj is None:
                raise ValueError(f"Premise ID {p} not found in statements.")
            a_obj.premise.append(p_obj)
        a_obj.conclusion = next((s for s in statements if s.id == int(a["conclusion"])), None)
        if a_obj.conclusion is None:
            raise ValueError(f"Conclusion ID {a['conclusion']} not found in statements.")
        a_obj.id = a["id"]
        a_obj.desc = a.get("desc", "")
        arguments.append(a_obj)
    return GraphData(statements=statements, arguments=arguments)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datafile", type=str, required=True, help="Path to JSON file with statements and arguments.")
    args = parser.parse_args()
    data = json.loads(Path(args.datafile).read_text())

    graphdata = process_json(args.datafile)

    embedder = SentenceTransformerEmbeddings()  # or None

    llm = openai_llm_call_factory(model="gpt-4o-mini")
    hi_statements, hi_args, answer = query_graph(
        "I believe in helping my local community more than distant causes. What arguments and evidence support this?",
        graphdata.statements,
        graphdata.arguments, 
        llm_call=llm,
        embedder=embedder,
        prefilter=True,
    )
    print("Highlighted Statements:", [s.id for s in hi_statements])
    print("Highlighted Arguments:", [a.id for a in hi_args])
    print("Answer:", answer)

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
from query_traverse_graph import *

# --- Your schema ---
from schema import Statement, Argument, Artifact, Quote, GraphData


# =========================
# 6) General query via LLM (planner + executor)
# =========================

from dataclasses import dataclass
from typing import Any, Dict, Optional, Callable

@dataclass
class _LLMPlan:
    mode: str                              # "supports" | "derive" | "answer"
    target_statement_id: Optional[int]     # only for "supports"
    seed_statement_ids: List[int]          # only for "derive"
    highlight_statement_ids: List[int]     # optional suggestions from LLM
    highlight_argument_ids: List[int]      # optional suggestions from LLM
    answer: str
    notes: str

def _serialize_graph_for_llm(statements: List[Statement], arguments: List[Argument]) -> Dict[str, Any]:
    """Compact JSON the LLM can reason over (ids + short text, minimal metadata)."""
    stmts_json = []
    for s in statements:
        arts = []
        if getattr(s, "artifact", None):
            for a in s.artifact:
                arts.append({"id": a.id, "name": a.name, "author": a.author, "title": a.title, "year": a.year})
        cites = []
        if getattr(s, "citations", None):
            for q in s.citations:
                cites.append({"page": q.page, "text": q.text})
        stmts_json.append({"id": s.id, "text": s.statement, "artifacts": arts, "citations": cites})

    args_json = [
        {"id": a.id,
         "premise_ids": [p.id for p in a.premise],
         "conclusion_id": a.conclusion.id,
         "desc": a.desc}
        for a in arguments
    ]
    return {"statements": stmts_json, "arguments": args_json}

def _general_system_prompt() -> str:
    return (
        "You are a planner that decides how to answer a question using a DAG of statements and arguments.\n"
        "Graph semantics: each Argument has premise statement ids and one conclusion id; if all premises hold, the conclusion is supported.\n\n"
        "Choose ONE mode:\n"
        "- 'supports': the user asks what supports a particular belief/statement; pick exactly one target_statement_id.\n"
        "- 'derive': the user asks what follows from a set of beliefs/first principles; pick seed_statement_ids.\n"
        "- 'answer': the user asks something else; just answer and optionally suggest highlights.\n\n"
        "Then produce a short answer (<=6 sentences, neutral tone) and minimal highlight ids.\n"
        "Use ONLY ids present in the provided graph. Do NOT invent ids or content. Return STRICT JSON.\n"
        "Do not reveal chain-of-thought; the 'notes' field should reference ids only (e.g., 'S3,S7 via A12 -> S9')."
    )

def _general_schema_text() -> str:
    return """
Return JSON exactly like this (no prose before or after):

{
  "mode": "supports | derive | answer",
  "target_statement_id": 123,
  "seed_statement_ids": [1, 2],
  "highlight_statement_ids": [3, 4],
  "highlight_argument_ids": [10, 11],
  "answer": "short direct answer (<= 6 sentences).",
  "notes": "brief id-only justification"
}

Rules:
- If mode is "supports": set target_statement_id and leave seed_statement_ids as [].
- If mode is "derive": set seed_statement_ids (non-empty) and set target_statement_id to null.
- If mode is "answer": set both target_statement_id = null and seed_statement_ids = [].
"""

def _brace_json_extract(raw: str) -> str:
    """Extract first balanced JSON object from a model response using brace-counting."""
    s = raw.strip()
    if s.startswith("{") and s.endswith("}"):
        try:
            json.loads(s); return s
        except Exception:
            pass
    depth, start = 0, -1
    for i, ch in enumerate(s):
        if ch == "{":
            if depth == 0: start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start != -1:
                    cand = s[start:i+1]
                    try:
                        json.loads(cand); return cand
                    except Exception:
                        start = -1
                        continue
    raise ValueError("Could not locate JSON in model output.")

def _parse_plan(payload: str) -> _LLMPlan:
    obj = json.loads(payload)
    mode = (obj.get("mode") or "").strip().lower()
    tgt = obj.get("target_statement_id", None)
    if isinstance(tgt, str) and tgt.isdigit():
        tgt = int(tgt)
    seeds = obj.get("seed_statement_ids", []) or []
    seeds = [int(x) for x in seeds if isinstance(x, (int, str)) and str(x).isdigit()]
    hs = obj.get("highlight_statement_ids", []) or []
    hs = [int(x) for x in hs if isinstance(x, (int, str)) and str(x).isdigit()]
    ha = obj.get("highlight_argument_ids", []) or []
    ha = [int(x) for x in ha if isinstance(x, (int, str)) and str(x).isdigit()]
    ans = (obj.get("answer") or "").strip()
    notes = (obj.get("notes") or "").strip()
    return _LLMPlan(mode=mode,
                    target_statement_id=tgt if isinstance(tgt, int) else None,
                    seed_statement_ids=seeds,
                    highlight_statement_ids=hs,
                    highlight_argument_ids=ha,
                    answer=ans,
                    notes=notes)

def _prefilter_for_llm(user_query: str,
                       statements: List[Statement],
                       arguments: List[Argument],
                       embedder: Optional[EmbeddingBackend],
                       max_statements: int = 220,
                       max_arguments: int = 440) -> Tuple[List[Statement], List[Argument]]:
    """Optional payload reduction: keep top-K statements by similarity to the query, keep connected arguments."""
    if embedder is None or len(statements) <= max_statements:
        return statements, arguments
    texts = [s.statement for s in statements] + [user_query]
    vecs = embedder.embed_texts(texts)
    qv = vecs[-1]
    scored = [(_cosine(v, qv), i) for i, v in enumerate(vecs[:-1])]
    scored.sort(reverse=True)
    keep_idx = {i for _, i in scored[:max_statements]}
    kept_statements = [s for i, s in enumerate(statements) if i in keep_idx]
    kept_ids = {s.id for s in kept_statements}
    kept_arguments = [a for a in arguments
                      if a.conclusion.id in kept_ids or any(p.id in kept_ids for p in a.premise)]
    if len(kept_arguments) > max_arguments:
        kept_arguments = kept_arguments[:max_arguments]
    return kept_statements, kept_arguments

def general_query(
    graphdata: GraphData,
    user_query: str,
    *,
    llm_call: Optional[Callable[[List[Dict[str, str]]], str]] = None,
    embedder: Optional[EmbeddingBackend] = None,
    prefilter: bool = False 
) -> Tuple[str, GraphData]:
    """
    Decide how to answer a user query using the graph:
      - If the LLM selects 'supports': run query_supports on target_statement_id.
      - If the LLM selects 'derive': run query_derivable on seed_statement_ids.
      - Else 'answer': just return the LLM's answer and its suggested highlights.

    Returns:
        (answer_string, highlighted_statements, highlighted_arguments)
    """
    g = Graph(graphdata)

    # Build (optionally filtered) context for the LLM
    all_statements = list(g.statements.values())
    all_arguments = g.arguments
    stmts_ctx, args_ctx = (all_statements, all_arguments)
    if prefilter:
        stmts_ctx, args_ctx = _prefilter_for_llm(user_query, all_statements, all_arguments, embedder)

    graph_json = _serialize_graph_for_llm(stmts_ctx, args_ctx)
    messages = [
        {"role": "system", "content": _general_system_prompt()},
        {"role": "user", "content": f"QUERY:\n{user_query}\n\nGRAPH:\n{json.dumps(graph_json, ensure_ascii=False)}\n\n{_general_schema_text()}"}
    ]

    # Call LLM (use provided callable or OpenAI fallback)
    if llm_call is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("general_query: OPENAI_API_KEY not set and no llm_call provided.")
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

    # Parse plan (with one repair attempt if needed)
    try:
        plan = _parse_plan(_brace_json_extract(raw))
    except Exception:
        repair_msgs = messages + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": "Your output was not valid JSON. Please return exactly one JSON object matching the schema."}
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
            plan = _parse_plan(_brace_json_extract(retry.choices[0].message.content or ""))
        else:
            plan = _parse_plan(_brace_json_extract(llm_call(repair_msgs)))

    # Execute plan
    mode = plan.mode
    answer = plan.answer or ""

    # helpers to map ids -> objects from the FULL graph (not the filtered one)
    stmt_by_id = g.statements
    arg_by_id = {a.id: a for a in g.arguments}

    if mode == "supports" and plan.target_statement_id is not None and plan.target_statement_id in stmt_by_id:
        res = query_supports(g, statement_id=plan.target_statement_id)
        hi_statements = res["all_supporting_statements"]
        hi_arguments = res["supporting_arguments"]
        # If the LLM didn't provide an answer, create a concise one.
        if not answer:
            answer = (
                f"Showing supporting chain for S{plan.target_statement_id}: "
                f"{len(hi_arguments)} argument(s), {len(hi_statements)} statement(s) involved."
            )
        ret_graph_data = GraphData(statements=hi_statements, arguments=hi_arguments)
        return answer, ret_graph_data

    if mode == "derive" and plan.seed_statement_ids:
        seeds = [sid for sid in plan.seed_statement_ids if sid in stmt_by_id]
        if not seeds and all_statements:
            # Fallback: pick nearest statement to the query as a single seed
            nearest = find_nearest_statements(all_statements, user_query, embedder)[:1] if embedder else []
            seeds = [nearest[0].statement.id] if nearest else []
        res = query_derivable(g, seed_statement_ids=seeds)
        new_conclusions: List[Statement] = res["new_conclusions"]
        proof_steps: List[ProofStep] = res["proof"]
        arg_ids_used = {ps.argument_id for ps in proof_steps}
        hi_arguments = [arg_by_id[aid] for aid in arg_ids_used if aid in arg_by_id]
        # Highlight seeds + new conclusions (minimal, legible)
        seed_objs = [stmt_by_id[sid] for sid in seeds if sid in stmt_by_id]
        hi_statements = seed_objs + new_conclusions
        if not answer:
            answer = (
                f"From seeds {sorted(seeds)}, derived {len(new_conclusions)} new statement(s) "
                f"via {len(hi_arguments)} argument(s)."
            )
        ret_graph_data = GraphData(statements=hi_statements, arguments=hi_arguments)
        return answer, ret_graph_data

    # Default: plain answer mode. Use LLM highlights if valid.
    hi_statements = [stmt_by_id[sid] for sid in plan.highlight_statement_ids if sid in stmt_by_id]
    hi_arguments = [arg_by_id[aid] for aid in plan.highlight_argument_ids if aid in arg_by_id]
    # If the model selected neither mode properly and gave no highlights, try a tiny heuristic: nearest statements.
    if not hi_statements and embedder is not None and all_statements:
        for m in find_nearest_statements(all_statements, user_query, embedder, top_k=5):
            hi_statements.append(m.statement)
    if not answer:
        answer = "I highlighted the most relevant parts of the graph to your question."

    ret_graph_data = GraphData(statements=hi_statements, arguments=hi_arguments)

    return answer, ret_graph_data



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run example queries on a statements/arguments dataset.")
    parser.add_argument("--datafile", type=str, required=True, help="Path to JSON file with statements and arguments.")
    args = parser.parse_args()

    graphdata = process_json(args.datafile)

    answer, ret_graph_data = general_query(
        graphdata,
        user_query="What should I do if I believe that we ought to help everyone in needs",
    )
    print(answer)
    print("Highlighted Statements:")
    for s in ret_graph_data.statements:
        print(f"  S{s.id}: {s.statement}")
    print("Highlighted Arguments:")
    for a in ret_graph_data.arguments:
        print(f"  A{a.id}: {a.desc}")

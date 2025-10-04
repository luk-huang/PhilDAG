import json
import os
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from DAG import Edge, Node, DAG

try:
    from pydantic import BaseModel, Field, ValidationError, conlist, confloat
    HAVE_PYDANTIC = True
except ImportError:
    HAVE_PYDANTIC = False

# -------------------------
# Configuration & Utilities
# -------------------------

@dataclass
class BuildConfig:
    window_size: Optional[int] = None  # None -> consider ALL prior nodes
    min_edge_confidence: float = 0.35
    consider_prompt_A: bool = True
    model_name: str = "gpt-4o-mini"
    dry_run: bool = False
    temperature: float = 0.2
    max_alternatives: int = 8

    # NEW: scoring-based shortlist to break adjacency bias
    scoring_threshold: float = 0.35    # keep priors with score >= this
    scoring_top_k: int = 12            # always keep top-K even if below threshold
    randomize_shortlist_order: bool = False  # set True to reduce positional bias in display

# -------------------------
# LLM client (pluggable)
# -------------------------

class AIClient:
    """
    Replace 'complete' method with your provider of choice.
    Example shows OpenAI's Chat Completions if OPENAI_API_KEY is set,
    otherwise falls back to a simple stub (dry-run).
    """
    def __init__(self, cfg: BuildConfig):
        self.cfg = cfg
        self._openai = None
        key = os.getenv("OPENAI_API_KEY")
        if key:
            try:
                from openai import OpenAI
                self._openai = OpenAI(api_key=key)
            except Exception:
                self._openai = None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8),
           retry=retry_if_exception_type(Exception))
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        if self.cfg.dry_run or self._openai is None:
            # Simple deterministic stub; always returns "premise" for Prompt A and "no edges" for Prompt B
            if '"is_premise"' in user_prompt:
                return json.dumps({"is_premise": True, "confidence": 0.66, "rationale_1_sentence": "Appears axiomatic/standalone."})
            return json.dumps({"edges": [], "global_confidence": 0.5, "notes": "dry-run"})
        # Real call
        resp = self._openai.chat.completions.create(
            model=self.cfg.model_name.split("/", 1)[-1],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.cfg.temperature,
        )
        return resp.choices[0].message.content

# -------------------------
# JSON Schemas (strict)
# -------------------------

if HAVE_PYDANTIC:
    class PromptAResp(BaseModel):
        is_premise: bool
        confidence: confloat(ge=0.0, le=1.0)  # type: ignore
        rationale_1_sentence: str

    class EdgeItem(BaseModel):
        from_ids: conlist(int)  # type: ignore
        tag: str
        confidence: confloat(ge=0.0, le=1.0)  # type: ignore
        note: str

    class PromptBResp(BaseModel):
        edges: List[EdgeItem]
        global_confidence: confloat(ge=0.0, le=1.0)  # type: ignore
        notes: str

# -------------------------
# Prompt templates
# -------------------------

PROMPT_S_SYSTEM = "You are a systematic argument screener. Answer ONLY in strict JSON."
PROMPT_B_SYSTEM = "You are a careful argument graph builder. Answer ONLY in strict JSON."

def prompt_S(target_id: int, target_text: str, prior_pairs: List[Tuple[int, str]]) -> str:
    prior_block = "\n".join([f"{pid}: {ptext}" for pid, ptext in prior_pairs])
    return f"""You are evaluating SUPPORT POTENTIAL for a target philosophical claim.
You MUST rate EVERY prior node (id < target_id) on whether it helps justify the target.

CRITICAL: Do NOT assume adjacency. Nearness in text is NOT evidence.
Older nodes can be decisive; newer ones can be irrelevant.

Output STRICTLY this JSON (no extra text):
{{
  "scores": [
    {{
      "id": <int>,
      "support_strength": <number 0..1>,
      "role": "<one of: Logic, Definition, Empirical, Analogy, Authority, Causal, Semantic, Reductio, Example, Other>",
      "why_<=12w": "<=12 words reason"
    }}
  ]
}}

INPUT:
- target_id: {target_id}
- target_text: \"\"\"{target_text}\"\"\"
- prior_nodes: [
{prior_block}
]"""

# --- Update Prompt B (edges) to use SHORTLIST and anti-adjacency guidance ---

def prompt_B_from_shortlist(target_id: int, target_text: str, shortlist_items: List[Tuple[int, float, str, str]]) -> str:
    # shortlist_items: List[(id, score, role, text)]
    if shortlist_items:
        disp = "\n".join([f"{pid} | support_strength={score:.2f} | role={role} | {txt}"
                          for (pid, score, role, txt) in shortlist_items])
    else:
        disp = "(empty)"
    return f"""Construct MINIMAL sufficient support sets for the target claim using ANY subset of the SHORTLIST below.
No adjacency bias: do NOT prefer nodes only because they are near in text.
If a singleton {{i-1}} is chosen, include a note justifying why earlier nodes are NOT needed.

Rules:
- Use only ids in SHORTLIST (all are < target_id).
- Each set must be MINIMAL: removing any member breaks sufficiency.
- Provide MULTIPLE alternative sets if plausible.
- If truly no support needed, return [].

NEGATIVE EXAMPLE (do NOT imitate):
- Bad: choosing only {{target_id-1}} with no rationale when older premises clearly matter.

Tags: ["Logic","Definition","Empirical","Analogy","Authority","Causal","Semantic","Reductio","Example","Other"].

Output STRICTLY this JSON:
{{
  "edges": [
    {{
      "from_ids": [<int>, ...],
      "tag": "<one tag>",
      "confidence": <0..1>,
      "note": "<=25 words why this set suffices>"
    }}
  ],
  "global_confidence": <0..1>,
  "notes": "<=50 words, overall caveats>"
}}

INPUT:
- target_id: {target_id}
- target_text: \"\"\"{target_text}\"\"\"
- SHORTLIST (candidates; order is arbitrary):
{disp}
"""

# -------------------------
# Core builder
# -------------------------

class DAGBuilder:
    def __init__(self, cfg: BuildConfig):
        self.cfg = cfg
        self.client = AIClient(cfg)

    def _slice_prior(self, all_pairs: List[Tuple[int, str]]) -> List[Tuple[int, str]]:
        # Keep ALL unless user explicitly sets a window
        if self.cfg.window_size is None:
            return all_pairs
        return all_pairs[-self.cfg.window_size:]

    def _score_priors(self, target_id: int, target_text: str, prior_pairs: List[Tuple[int, str]]):
        """Return list of (id, score, role, text) for ALL priors."""
        if not prior_pairs:
            return []
        up = prompt_S(target_id, target_text, prior_pairs)
        raw = self.client.complete(PROMPT_S_SYSTEM, up)
        try:
            data = json.loads(raw)
            out = []
            seen = set()
            for item in data.get("scores", []):
                pid = int(item.get("id"))
                if pid in seen:
                    continue
                seen.add(pid)
                score = float(item.get("support_strength", 0.0))
                role = str(item.get("role", "Other"))
                # fetch text
                txt = next((t for (iid, t) in prior_pairs if iid == pid), "")
                out.append((pid, score, role, txt))
            # sort by score desc, then id asc
            out.sort(key=lambda x: (-x[1], x[0]))
            # threshold + top-k
            kept = [r for r in out if r[1] >= self.cfg.scoring_threshold]
            if len(kept) < self.cfg.scoring_top_k:
                kept_ids = {k[0] for k in kept}
                for r in out:
                    if r[0] not in kept_ids:
                        kept.append(r)
                    if len(kept) >= self.cfg.scoring_top_k:
                        break
            if self.cfg.randomize_shortlist_order:
                import random
                random.shuffle(kept)
            return kept
        except Exception:
            # Fallback: keep last K and first K to diversify
            K = self.cfg.scoring_top_k
            head = prior_pairs[:K//2]
            tail = prior_pairs[-(K - len(head)):] if len(prior_pairs) > K//2 else []
            mix = head + tail
            return [(pid, 0.4, "Other", txt) for (pid, txt) in mix]

    def _propose_edges_from_shortlist(self, target_id: int, target_text: str, shortlist):
        up = prompt_B_from_shortlist(target_id, target_text, shortlist)
        raw = self.client.complete(PROMPT_B_SYSTEM, up)
        try:
            data = json.loads(raw)
            edges = []
            for e in data.get("edges", [])[: self.cfg.max_alternatives]:
                conf = float(e.get("confidence", 0.0))
                if conf < self.cfg.min_edge_confidence:
                    continue
                from_ids = sorted(set(int(x) for x in e.get("from_ids", [])))
                tag = str(e.get("tag", "Logic"))
                note = str(e.get("note", ""))
                edges.append({"from_ids": from_ids, "tag": tag, "confidence": conf, "note": note})
            return edges
        except Exception:
            return []
        

    def _decide_premise(self, target_id: int, target_text: str, prior_pairs: List[Tuple[int, str]]) -> Tuple[bool, float]:
        up = prompt_S(target_id, target_text, prior_pairs)
        raw = self.client.complete(PROMPT_S_SYSTEM, up)
        try:
            data = json.loads(raw)
            if HAVE_PYDANTIC:
                parsed = PromptAResp(**data)
                return parsed.is_premise, float(parsed.confidence)
            # Fallback: minimal checks
            return bool(data.get("is_premise", False)), float(data.get("confidence", 0.5))
        except Exception:
            # If malformed, default to "needs support"
            return False, 0.0

    def build_from_lines(self, lines: List[str]) -> DAG:
        text_list = [ln.strip() for ln in lines]
        dag = DAG(n=len(text_list), text_list=text_list)

        for i, text in enumerate(text_list):
            prior_pairs_all = [(pid, text_list[pid]) for pid in range(i)]
            prior_pairs = self._slice_prior(prior_pairs_all)

            if self.cfg.consider_prompt_A and prior_pairs:
                is_premise, _ = self._decide_premise(i, text, prior_pairs)
                if is_premise:
                    continue

            # NEW: global scoring of ALL priors
            shortlist = self._score_priors(i, text, prior_pairs)
            # Build edges from that shortlist (can be any subset; no adjacency bias)
            edges = self._propose_edges_from_shortlist(i, text, shortlist)

            # Add each minimal sufficient set as an Edge
            valid_prior_ids = {pid for (pid, _, _, _) in shortlist}
            for e in edges:
                # enforce legality and within range
                from_ids = [pid for pid in e["from_ids"] if 0 <= pid < i and pid in valid_prior_ids]
                if not from_ids and e["from_ids"]:
                    # model tried to use out-of-shortlist; skip this set
                    continue
                from_nodes = [dag.dag[j] for j in from_ids]
                dag.add_edge(Edge(from_node_list=from_nodes, to_node=i, tag=e["tag"]))

        return dag

# -------------------------
# I/O helpers and Graphviz
# -------------------------

def read_txt(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f]

def export_graphviz(dag: DAG, out_path: str = "argument_dag.dot"):
    """
    Produces a Graphviz DOT file with alternative sufficient sets shown as distinct edges.
    """
    lines = ["digraph ArgumentDAG {", '  rankdir=LR;', '  node [shape=box];']
    for node in dag.dag:
        label = f"{node.id}: {node.text.replace('\"','\\\"')}"
        lines.append(f'  n{node.id} [label="{label}"];')

    # For each clause (edge), draw edges from each from_node to to_node; annotate with tag
    for node in dag.dag:
        for clause_idx, edge in enumerate(node.clauses):
            for fn in edge.from_node_list:
                # Use a distinct edge color/style per clause index modulo a few styles
                style = ["solid","dashed","dotted","bold"][clause_idx % 4]
                lines.append(f'  n{fn.id} -> n{edge.to_node} [label="{edge.tag}", style={style}];')

    lines.append("}")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return out_path

# -------------------------
# Example CLI usage
# -------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to .txt; each line is a node (argument/claim).")
    parser.add_argument("--window-size", type=int, default=0, help="If >0, limit to last K prior nodes for context.")
    parser.add_argument("--no-prompt-A", action="store_true", help="Disable premise-vs-supported classification; always attempt edges.")
    parser.add_argument("--dry-run", action="store_true", help="Do not call an API; return empty edges/premises.")
    parser.add_argument("--dot", type=str, default="argument_dag.dot", help="Output .dot filepath")
    args = parser.parse_args()

    cfg = BuildConfig(
        window_size=None if args.window_size <= 0 else args.window_size,
        consider_prompt_A=not args.no_prompt_A,
        dry_run=args.dry_run,
    )
    builder = DAGBuilder(cfg)

    lines = read_txt(args.input)
    dag = builder.build_from_lines(lines)
    dotfile = export_graphviz(dag, args.dot)
    print(f"Wrote Graphviz DOT to: {dotfile}")

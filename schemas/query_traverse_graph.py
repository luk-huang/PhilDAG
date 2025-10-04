# reasoning.py (schema-updated)

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Iterable, Optional, Callable
from collections import defaultdict, deque
import math
import argparse
import json
from pathlib import Path

# ---- Import your schema ----
from analysis import Quote, Artifact, Statement, Argument

# =========================
# Core graph construction
# =========================

@dataclass(frozen=True)
class ProofStep:
    """One application of an argument in a proof."""
    argument_id: int
    premise_ids: Tuple[int, ...]
    conclusion_id: int

class Graph:
    """
    Thin index over (statements, arguments) with helpers for:
      - supporting arguments for a statement (backward)
      - forward chaining (derive statements from seeds)
      - contradiction checks
      - nearest-statement retrieval (pluggable embeddings)
    """
    def __init__(self, statements: List[Statement], arguments: List[Argument]):
        self.statements: Dict[int, Statement] = {s.id: s for s in statements}
        self.arguments: List[Argument] = arguments

        # Map: conclusion -> [arguments concluding it]
        self.by_conclusion: Dict[int, List[Argument]] = defaultdict(list)
        for a in arguments:
            self.by_conclusion[a.conclusion.id].append(a)

        # Map: statement -> arguments where it appears as a premise
        self.by_premise: Dict[int, List[Argument]] = defaultdict(list)
        for a in arguments:
            for p in a.premise:
                self.by_premise[p.id].append(a)

        # Sanity: detect obvious cycles (should be DAG)
        self._assert_acyclic()

    # ---------- DAG check ----------
    def _assert_acyclic(self) -> None:
        indeg = defaultdict(int)
        adj = defaultdict(list)
        for a in self.arguments:
            for p in a.premise:
                indeg[a.conclusion.id] += 1
                adj[p.id].append(a.conclusion.id)

        q = deque([sid for sid in self.statements if indeg[sid] == 0])
        seen = 0
        while q:
            u = q.popleft()
            seen += 1
            for v in adj[u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)
        if seen != len(self.statements):
            raise ValueError("Cycle detected: graph must be a DAG.")

    # =========================
    # 1) Backward: supports for a statement
    # =========================
    def supporting_arguments(self, statement_id: int) -> Tuple[List[Statement], List[Argument]]:
        """
        Returns all statements and arguments that (recursively) support statement_id.
        Topologically traverses premises via DFS.
        """
        visited: Set[int] = set()
        stmts_acc: List[Statement] = []
        args_acc: List[Argument] = []

        def dfs(sid: int) -> None:
            if sid in visited:
                return
            visited.add(sid)
            if sid in self.by_conclusion:
                for arg in self.by_conclusion[sid]:
                    args_acc.append(arg)
                    for prem in arg.premise:
                        dfs(prem.id)
            # collect after exploring
            stmts_acc.append(self.statements[sid])

        dfs(statement_id)
        # Put the target statement first for convenience
        stmts_acc.sort(key=lambda s: (s.id != statement_id, s.id))
        return stmts_acc, args_acc

    # Minimal support sets (optional): returns arguments whose premises are all "axioms"
    def minimal_support_sets(self, statement_id: int, axioms: Set[int]) -> List[List[int]]:
        """
        Finds premise-id lists that, if assumed, are sufficient to prove statement_id
        without needing other derived statements. Exponential in worst case.
        """
        results: List[List[int]] = []

        def rec(sid: int) -> List[List[int]]:
            if sid in axioms or sid not in self.by_conclusion:
                return [[sid]] if sid in axioms else [[]]  # [] means "already given/non-axiom leaf"
            combos: List[List[int]] = []
            for arg in self.by_conclusion[sid]:
                lists_per_prem = [rec(p.id) for p in arg.premise]
                # flatten cartesian product
                def product(acc: List[List[int]], nxt: List[List[int]]) -> List[List[int]]:
                    if not acc:
                        return [x[:] for x in nxt]
                    out = []
                    for a in acc:
                        for b in nxt:
                            out.append(a + b)
                    return out
                prod: List[List[int]] = []
                for lst in lists_per_prem:
                    prod = product(prod, lst)
                for s in prod:
                    uniq = sorted(set(x for x in s if x in axioms))
                    combos.append(uniq)
            # dedup
            seen = set()
            dedup = []
            for cset in combos:
                t = tuple(cset)
                if t not in seen:
                    seen.add(t)
                    dedup.append(cset)
            return dedup

        for cset in rec(statement_id):
            if cset:
                results.append(cset)
        return results

    # =========================
    # 2) Forward: derive from first principles
    # =========================
    def derive_from(self, true_statement_ids: Iterable[int]) -> Tuple[Set[int], List[ProofStep]]:
        """
        Given a set of believed statements (first principles), forward-chain to closure.
        Each Argument fires iff all its premises are currently true.
        Returns (all_true_ids, proof_steps_used).
        """
        true_set: Set[int] = set(true_statement_ids)
        fired: List[ProofStep] = []

        remaining: Dict[int, int] = {}
        prem_sets: Dict[int, Set[int]] = {}
        concluded: Set[int] = set()

        for a in self.arguments:
            p_ids = {p.id for p in a.premise}
            prem_sets[a.id] = p_ids
            remaining[a.id] = len(p_ids - true_set)

        changed = True
        while changed:
            changed = False
            for a in self.arguments:
                if a.conclusion.id in true_set:
                    continue
                if remaining[a.id] == 0:
                    # Fire argument
                    true_set.add(a.conclusion.id)
                    fired.append(ProofStep(
                        argument_id=a.id,
                        premise_ids=tuple(sorted(prem_sets[a.id])),
                        conclusion_id=a.conclusion.id
                    ))
                    concluded.add(a.conclusion.id)
                    changed = True
                    # Update other arguments that depend on this conclusion
                    for dep in self.by_premise.get(a.conclusion.id, []):
                        if remaining[dep.id] > 0:
                            remaining[dep.id] -= 1
        return true_set, fired

    # =========================
    # 3) Contradictions
    # =========================
    def contradictions_explicit(
        self, believed_ids: Iterable[int], contradiction_pairs: Set[Tuple[int, int]]
    ) -> List[Tuple[Statement, Statement]]:
        """
        If you maintain an explicit set of contradictory pairs (idA, idB),
        return those present in the believed set.
        """
        s = set(believed_ids)
        out: List[Tuple[Statement, Statement]] = []
        for a, b in contradiction_pairs:
            if a in s and b in s:
                out.append((self.statements[a], self.statements[b]))
        return out

    def contradictions_nli(
        self,
        believed_ids: Iterable[int],
        is_contradiction: Callable[[str, str], float],
        threshold: float = 0.8
    ) -> List[Tuple[Statement, Statement, float]]:
        """
        Uses a provided NLI scorer: is_contradiction(textA, textB)->score in [0,1].
        Returns all pairs with score >= threshold.
        Note: O(n^2) on the number of believed statements.
        """
        ids = list(believed_ids)
        out: List[Tuple[Statement, Statement, float]] = []
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                si, sj = self.statements[ids[i]], self.statements[ids[j]]
                s = is_contradiction(si.statement, sj.statement)
                if s >= threshold:
                    out.append((si, sj, s))
        return out

# =========================================
# Convenience: robust cosine + FAISS fallback
# =========================================
def _cosine(u, v) -> float:
    num = sum(a*b for a, b in zip(u, v))
    du = math.sqrt(sum(a*a for a in u))
    dv = math.sqrt(sum(b*b for b in v))
    return 0.0 if du == 0 or dv == 0 else num / (du * dv)

# =========================================
# 4) Nearest-statement search (pluggable backends)
# =========================================
class EmbeddingBackend:
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError

class OpenAIEmbeddings(EmbeddingBackend):
    """
    Requires `openai` package and OPENAI_API_KEY in env.
    Replace model name as desired (e.g., 'text-embedding-3-large')
    """
    def __init__(self, model: str = "text-embedding-3-small"):
        import openai  # type: ignore
        self._openai = openai
        self.model = model

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        resp = self._openai.Embeddings.create(model=self.model, input=texts)  # old SDK
        return [d["embedding"] for d in resp["data"]]

class SentenceTransformerEmbeddings(EmbeddingBackend):
    """
    Offline-friendly using sentence-transformers.
    e.g., model_name='sentence-transformers/all-MiniLM-L6-v2'
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # type: ignore
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, normalize_embeddings=False).tolist()

@dataclass
class StatementMatch:
    statement: Statement
    score: float
    idx: int

def find_nearest_statements(
    statements: List[Statement],
    query: str,
    embedder: EmbeddingBackend,
    top_k: int = 5
) -> List[StatementMatch]:
    """
    Embeds all statement texts + query and returns the top_k most similar.
    If FAISS is available, uses it; otherwise, falls back to pure Python cosine.
    """
    import importlib
    texts = [s.statement for s in statements]
    embs = embedder.embed_texts(texts + [query])
    stmt_vecs = embs[:-1]
    qvec = embs[-1]

    # Try FAISS
    if importlib.util.find_spec("faiss") is not None:
        import faiss  # type: ignore
        import numpy as np  # type: ignore
        A = np.array(stmt_vecs, dtype="float32")
        q = np.array([qvec], dtype="float32")
        def l2norm(X):
            norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
            return X / norms
        A_n = l2norm(A)
        q_n = l2norm(q)
        index = faiss.IndexFlatIP(A.shape[1])
        index.add(A_n)
        D, I = index.search(q_n, min(top_k, len(statements)))
        out = []
        for idx, score in zip(I[0].tolist(), D[0].tolist()):
            out.append(StatementMatch(statement=statements[idx], score=float(score), idx=idx))
        return out

    # Fallback: compute cosine in Python
    scored = [(i, _cosine(vec, qvec)) for i, vec in enumerate(stmt_vecs)]
    scored.sort(key=lambda t: t[1], reverse=True)
    return [StatementMatch(statement=statements[i], score=s, idx=i) for i, s in scored[:top_k]]

# =========================================
# 5) Example user-query helpers
# =========================================

def query_supports(g: Graph, statement_id: int) -> Dict[str, object]:
    """
    'If I believe this statement, what arguments support it?'
    Returns:
      - all_supporting_statements: List[Statement]
      - supporting_arguments: List[Argument]
      - minimal_axiom_sets: List[List[int]]  (optional, assumes axioms are leaves)
    """
    stmts, args = g.supporting_arguments(statement_id)
    # Heuristic axioms: statements that never appear as conclusions
    non_conclusions = {sid for sid in g.statements if sid not in g.by_conclusion}
    minimal_sets = g.minimal_support_sets(statement_id, non_conclusions)
    return {
        "all_supporting_statements": stmts,
        "supporting_arguments": args,
        "minimal_axiom_sets": minimal_sets,
    }

def query_derivable(g: Graph, seed_statement_ids: Iterable[int]) -> Dict[str, object]:
    """
    'What statements can I derive from believing certain first principles?'
    Returns:
      - all_true_ids (closure)
      - new_conclusions (true minus seeds)
      - proof (ordered list of ProofStep)
    """
    seeds = set(seed_statement_ids)
    closure, proof = g.derive_from(seeds)
    new_conclusions = [g.statements[sid] for sid in closure - seeds]
    return {
        "all_true_ids": sorted(list(closure)),
        "new_conclusions": new_conclusions,
        "proof": proof,
    }

def query_support_from_usertext(g: Graph, text: str, p_nec: float = 0.5):
    """
    Get the statements that are closest to the user text,
    and then figure out which statements are necessary to support them
    """

    embedder = SentenceTransformerEmbeddings("sentence-transformers/all-MiniLM-L6-v2")
    match = find_nearest_statements(list(g.statements.values()), text, embedder=embedder, top_k = 1)[0]

    if match.score < p_nec:
        return {"all_supporting_statements": [], "supporting_arguments": [], "minimal_axiom_sets": []}
    return query_supports(g, match.statement.id), match
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run example queries on a statements/arguments dataset.")
    parser.add_argument("--datafile", type=str, required=True, help="Path to JSON file with statements and arguments.")
    args = parser.parse_args()


    data = json.loads(Path(args.datafile).read_text())
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

    g = Graph(statements, arguments)

    # 1) If I believe in statement X, what supports it?
    resp = query_supports(g, statement_id=10)
    print([s.id for s in resp["all_supporting_statements"]])
    print([a.id for a in resp["supporting_arguments"]])
    print("Minimal axiom sets (IDs):", resp["minimal_axiom_sets"])

    # 2) What statements can I derive from first principles?
    closure = query_derivable(g, seed_statement_ids=[1, 2, 5])
    print("All true:", closure["all_true_ids"])
    print("New conclusions:", [s.id for s in closure["new_conclusions"]])
    for step in closure["proof"]:
        print("Fired", step.argument_id, ":", step.premise_ids, "->", step.conclusion_id)

    # 4) Nearest statement to a text query
    """
    embedder = SentenceTransformerEmbeddings("sentence-transformers/all-MiniLM-L6-v2")
    matches = find_nearest_statements(statements, query="free will is compatible with determinism", embedder=embedder, top_k=5)
    for m in matches:
        print(m.statement.id, m.score, m.statement.statement)
    """

    # 5) Example: support from user text
    resp, match = query_support_from_usertext(g, text="we are one community so it is just as easy to help distant people as locals", p_nec=0.2)
    print("Base claim match:", match.statement.id, match.score, match.statement.statement)
    print("Support for user text:")
    for s in resp["all_supporting_statements"]:
        print(s.id, s.statement)

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Iterable, Optional, Callable
from collections import defaultdict, deque
import math
import argparse

# ---- Import your schema ----
from analysis import Quote, Artifact, Claim, Argument

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
    Thin index over (claims, arguments) with helpers for:
      - supporting arguments for a claim (backward)
      - forward chaining (derive claims from seeds)
      - contradiction checks
      - nearest-claim retrieval (pluggable embeddings)
    """
    def __init__(self, claims: List[Claim], arguments: List[Argument]):
        self.claims: Dict[int, Claim] = {c.id: c for c in claims}
        self.arguments: List[Argument] = arguments

        # Map: conclusion -> [arguments concluding it]
        self.by_conclusion: Dict[int, List[Argument]] = defaultdict(list)
        for a in arguments:
            self.by_conclusion[a.conclusion.id].append(a)

        # Map: claim -> arguments where it appears as a premise
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

        q = deque([cid for cid in self.claims if indeg[cid] == 0])
        seen = 0
        while q:
            u = q.popleft()
            seen += 1
            for v in adj[u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)
        if seen != len(self.claims):
            raise ValueError("Cycle detected: graph must be a DAG.")

    # =========================
    # 1) Backward: supports for a claim
    # =========================
    def supporting_arguments(self, claim_id: int) -> Tuple[List[Claim], List[Argument]]:
        """
        Returns all claims and arguments that (recursively) support claim_id.
        Topologically traverses premises via DFS.
        """
        visited: Set[int] = set()
        claims_acc: List[Claim] = []
        args_acc: List[Argument] = []

        def dfs(cid: int) -> None:
            if cid in visited:
                return
            visited.add(cid)
            if cid in self.by_conclusion:
                for arg in self.by_conclusion[cid]:
                    args_acc.append(arg)
                    for prem in arg.premise:
                        dfs(prem.id)
            # push after exploring (collect all unique once)
            claims_acc.append(self.claims[cid])

        dfs(claim_id)
        # Put the target claim first for convenience
        claims_acc.sort(key=lambda c: (c.id != claim_id, c.id))
        return claims_acc, args_acc

    # Minimal support sets (optional): returns arguments whose premises are all "axioms"
    def minimal_support_sets(self, claim_id: int, axioms: Set[int]) -> List[List[int]]:
        """
        Finds premise-id lists that, if assumed, are sufficient to prove claim_id
        without needing other derived claims. This is exponential in worst case,
        so use on modest subgraphs.
        """
        results: List[List[int]] = []

        def rec(cid: int) -> List[List[int]]:
            if cid in axioms or cid not in self.by_conclusion:
                return [[cid]] if cid in axioms else [[]]  # empty list means "already given"
            combos: List[List[int]] = []
            for arg in self.by_conclusion[cid]:
                # Cartesian product of supports for each premise
                lists_per_prem = [rec(p.id) for p in arg.premise]
                if any(len(lst) == 0 for lst in lists_per_prem):
                    # if any premise returns [], it means that premise is already given (non-axiom leaf)
                    pass
                # flatten product
                def product(acc: List[List[int]], nxt: List[List[int]]) -> List[List[int]]:
                    if not acc:
                        return [x[:] for x in nxt]
                    out = []
                    for a in acc:
                        for b in nxt:
                            out.append(a + b)
                    return out
                prod = []
                for lst in lists_per_prem:
                    prod = product(prod, lst)
                # Dedup ids inside each set and keep as list
                for s in prod:
                    uniq = sorted(set(x for x in s if x in axioms))
                    combos.append(uniq)
            # Dedup by tuple
            seen = set()
            dedup = []
            for cset in combos:
                t = tuple(cset)
                if t not in seen:
                    seen.add(t)
                    dedup.append(cset)
            return dedup

        for cset in rec(claim_id):
            if cset:  # non-empty actual axiom set
                results.append(cset)
        return results

    # =========================
    # 2) Forward: derive from first principles
    # =========================
    def derive_from(self, true_claim_ids: Iterable[int]) -> Tuple[Set[int], List[ProofStep]]:
        """
        Given a set of believed claims (first principles), forward-chain to closure.
        Each Argument fires iff all its premises are currently true.
        Returns (all_true_ids, proof_steps_used).
        """
        true_set: Set[int] = set(true_claim_ids)
        fired: List[ProofStep] = []

        # Track how many premises remain for each argument
        remaining: Dict[int, int] = {}
        prem_sets: Dict[int, Set[int]] = {}
        concluded: Set[int] = set()

        for a in self.arguments:
            p_ids = {p.id for p in a.premise}
            prem_sets[a.id] = p_ids
            remaining[a.id] = len(p_ids - true_set)

        # Worklist seeded by any arg already satisfied
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
                    # Update others that use this conclusion as a premise
                    for dep in self.by_premise.get(a.conclusion.id, []):
                        if remaining[dep.id] > 0:
                            remaining[dep.id] -= 1
        return true_set, fired

    # =========================
    # 3) Contradictions
    # =========================
    def contradictions_explicit(
        self, believed_ids: Iterable[int], contradiction_pairs: Set[Tuple[int, int]]
    ) -> List[Tuple[Claim, Claim]]:
        """
        If you maintain an explicit set of contradictory pairs (idA, idB), 
        return those present in the believed set.
        """
        s = set(believed_ids)
        out: List[Tuple[Claim, Claim]] = []
        for a, b in contradiction_pairs:
            if a in s and b in s:
                out.append((self.claims[a], self.claims[b]))
        return out

    def contradictions_nli(
        self,
        believed_ids: Iterable[int],
        is_contradiction: Callable[[str, str], float],
        threshold: float = 0.8
    ) -> List[Tuple[Claim, Claim, float]]:
        """
        Uses a provided NLI scorer: is_contradiction(textA, textB)->score in [0,1].
        Returns all pairs with score >= threshold.
        Note: O(n^2) on the number of believed claims.
        """
        ids = list(believed_ids)
        out: List[Tuple[Claim, Claim, float]] = []
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                ci, cj = self.claims[ids[i]], self.claims[ids[j]]
                s = is_contradiction(ci.desc, cj.desc)
                if s >= threshold:
                    out.append((ci, cj, s))
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
# 4) Nearest-claim search (pluggable backends)
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
        # Batched for efficiency
        resp = self._openai.Embeddings.create(model=self.model, input=texts)  # old SDK
        # If using new SDKs, adjust accordingly; keep simple for now:
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
class ClaimMatch:
    claim: Claim
    score: float
    idx: int

def find_nearest_claims(
    claims: List[Claim],
    query: str,
    embedder: EmbeddingBackend,
    top_k: int = 5
) -> List[ClaimMatch]:
    """
    Embeds all claim descriptions + query and returns the top_k most similar.
    If FAISS is available, uses it; otherwise, falls back to pure Python cosine.
    """
    import importlib
    descs = [c.desc for c in claims]
    embs = embedder.embed_texts(descs + [query])
    claim_vecs = embs[:-1]
    qvec = embs[-1]

    # Try FAISS
    if importlib.util.find_spec("faiss") is not None:
        import faiss  # type: ignore
        import numpy as np  # type: ignore
        # cosine similarity via inner product on normalized vectors
        A = np.array(claim_vecs, dtype="float32")
        q = np.array([qvec], dtype="float32")
        # normalize
        def l2norm(X):
            norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
            return X / norms
        A_n = l2norm(A)
        q_n = l2norm(q)
        index = faiss.IndexFlatIP(A.shape[1])
        index.add(A_n)
        D, I = index.search(q_n, min(top_k, len(claims)))
        out = []
        for rank, (idx, score) in enumerate(zip(I[0].tolist(), D[0].tolist())):
            out.append(ClaimMatch(claim=claims[idx], score=float(score), idx=idx))
        return out

    # Fallback: compute cosine in Python
    scored = [(i, _cosine(vec, qvec)) for i, vec in enumerate(claim_vecs)]
    scored.sort(key=lambda t: t[1], reverse=True)
    return [ClaimMatch(claim=claims[i], score=s, idx=i) for i, s in scored[:top_k]]

# =========================================
# 5) Example user-query helpers
# =========================================

def query_supports(g: Graph, claim_id: int) -> Dict[str, object]:
    """
    'If I believe this claim, what arguments support it?'
    Returns:
      - all_supporting_claims: List[Claim]
      - supporting_arguments: List[Argument]
      - minimal_axiom_sets: List[List[int]]  (optional, assumes axioms are leaves)
    """
    claims, args = g.supporting_arguments(claim_id)
    # Heuristic axioms: claims that never appear as conclusions
    non_conclusions = {cid for cid in g.claims if cid not in g.by_conclusion}
    minimal_sets = g.minimal_support_sets(claim_id, non_conclusions)
    return {
        "all_supporting_claims": claims,
        "supporting_arguments": args,
        "minimal_axiom_sets": minimal_sets,
    }

def query_derivable(g: Graph, seed_claim_ids: Iterable[int]) -> Dict[str, object]:
    """
    'What claims can I derive from believing certain first principles?'
    Returns:
      - all_true_ids (closure)
      - new_conclusions (true minus seeds)
      - proof (ordered list of ProofStep)
    """
    seeds = set(seed_claim_ids)
    closure, proof = g.derive_from(seeds)
    new_conclusions = [g.claims[cid] for cid in closure - seeds]
    return {
        "all_true_ids": sorted(list(closure)),
        "new_conclusions": new_conclusions,
        "proof": proof,
    }


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run example queries on a claims/arguments dataset.")
    parser.add_argument("--datafile", type=str, required=True, help="Path to JSON file with claims and arguments.")
    args = parser.parse_args()

    import json
    from pathlib import Path
    data = json.loads(Path(args.datafile).read_text())
    claims = [Claim(**c) for c in data["claims"]]
    arguments = [Argument(**a) for a in data["arguments"]]

    g = Graph(claims, arguments)

    # 1) If I believe in claim X, what supports it?
    resp = query_supports(g, claim_id=42)
    print([c.id for c in resp["all_supporting_claims"]])
    print([a.id for a in resp["supporting_arguments"]])
    print("Minimal axiom sets (IDs):", resp["minimal_axiom_sets"])

    # 2) What claims can I derive from first principles?
    closure = query_derivable(g, seed_claim_ids=[1, 2, 5])
    print("All true:", closure["all_true_ids"])
    print("New conclusions:", [c.id for c in closure["new_conclusions"]])
    for step in closure["proof"]:
        print("Fired", step.argument_id, ":", step.premise_ids, "->", step.conclusion_id)

    # 4) Nearest claim to a text query
    embedder = SentenceTransformerEmbeddings("sentence-transformers/all-MiniLM-L6-v2")
    matches = find_nearest_claims(claims, query="free will is compatible with determinism", embedder=embedder, top_k=5)
    for m in matches:
        print(m.claim.id, m.score, m.claim.desc)

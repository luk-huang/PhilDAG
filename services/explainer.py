import asyncio
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

import networkx as nx
from dotenv import load_dotenv
from google import genai
from google.genai import types

from schemas.schema import (
    Argument as SchemaArgument,
    Artifact as SchemaArtifact,
    GraphData,
    Quote as SchemaQuote,
    Statement as SchemaStatement,
)

load_dotenv()
client = genai.Client()

def _extract_text_from_response(response: types.GenerateContentResponse) -> str:
    if getattr(response, "text", None):
        return response.text

    parts: list[str] = []
    for candidate in getattr(response, "candidates", []) or []:
        content = getattr(candidate, "content", None)
        if not content:
            continue
        for part in getattr(content, "parts", []) or []:
            text = getattr(part, "text", None)
            if text:
                parts.append(text)
    return "\n".join(parts)


class MultiDocumentPhilosophyDAG:
    def __init__(
        self,
        file_paths: Optional[Sequence[Path]] = None,
        sample_size: int = 5,
        *,
        model: Optional[str] = None,
    ):
        self.sample_size = max(1, sample_size)
        self.model = model or os.getenv("GEMINI_EXPLAIN_MODEL", "gemini-2.5-flash")
        self.file_paths: List[Path] = []
        self._file_set: Set[Path] = set()
        self.statements = {}
        self.arguments = {}
        self.artifacts = {}
        self.graph = nx.DiGraph()
        self.next_statement_id = 1
        self.next_argument_id = 1
        self.next_artifact_id = 1
        self.statement_sources = {}
        self.argument_sources = {}
        self._uploaded_files: Dict[Path, object] = {}

        if file_paths:
            for path in file_paths:
                self.add_source(path)

    def add_source(self, path: Path) -> None:
        resolved = Path(path).resolve()
        if resolved in self._file_set:
            return
        if not resolved.is_file():
            raise FileNotFoundError(f"Source file not found: {resolved}")
        self.file_paths.append(resolved)
        self._file_set.add(resolved)

    def _sample_files(self) -> List[Path]:
        if not self.file_paths:
            return []
        count = min(self.sample_size, len(self.file_paths))
        return random.sample(self.file_paths, count)

    def _ensure_upload(self, path: Path):
        if path not in self._uploaded_files:
            self._uploaded_files[path] = client.files.upload(file=path)
        return self._uploaded_files[path]

    def build_prompt(self, sampled_files: Sequence[Path]):
        prompt = """
        You are analyzing multiple philosophical texts to build a unified Directed Acyclic Graph (DAG) of logical arguments.
        TASK: Extract MULTIPLE items from the texts below - either statements (claims) or arguments (justifications).
        Find connections between ideas across different texts when possible.
        KEY CONCEPTS:
        - A STATEMENT is any claim or assertion from the texts (unsubstantiated on its own)
        - An ARGUMENT connects premise statements to a conclusion statement with justification
        - Statements are the nodes; Arguments are the edges that connect them
        - Look for common axioms or claims across texts
        IMPORTANT RULES:
        1. Extract 5-10 items per response (statements or arguments)
        2. Statements should be atomic, precise, and concise claims
        3. Arguments can ONLY reference existing statement IDs as premises and conclusion
        4. When similar ideas appear in multiple texts, create linking arguments
        CURRENT STATEMENTS IN THE DAG:
        """
        if self.statements:
            for id in sorted(self.statements.keys()):
                stmt = self.statements[id]
                sources = self.statement_sources.get(id, set())
                source_str = f" [from: {', '.join(sources)}]" if sources else ""
                prompt += f"Statement {id}: \"{stmt['statement']}\"{source_str}\n"
        else:
            prompt += "[Empty - no statements yet]\n"

        prompt += "\nCURRENT ARGUMENTS:\n"
        if self.arguments:
            for id in sorted(self.arguments.keys()):
                arg = self.arguments[id]
                prompt += f"Argument {id}: {arg['premise']} → {arg['conclusion']}: {arg['desc']}\n"
        else:
            prompt += "[Empty - no arguments yet]\n"

        prompt += f"\nTotal statements: {len(self.statements)}, Total arguments: {len(self.arguments)}\n"
        prompt += "\nATTACHED PHILOSOPHICAL TEXTS:\n"

        for path in sampled_files:
            prompt += f"- Use the uploaded file named \"{path.name}\".\n"

        prompt += """---
        Extract 5-10 new items (statements or arguments) from these texts.
        Return JSON as {"items": [...]} where each item has:
        - "type": either "statement" or "argument"  
        - "data": the relevant data for that type
        - "source": filename(s) this item comes from (use the file names listed above)
        For statements:
        {"type": "statement", "data": {"statement": "The claim in clear modern language"}, "source": "filename.txt"}
        For arguments (can only use existing statement IDs):
        {"type": "argument", "data": {"premise_ids": [1, 2], "conclusion_id": 3, "desc": "Brief description of the reasoning"}, "source": "filename.txt"}
        IMPORTANT: Arguments can ONLY reference statement IDs that already exist.
        """
        return prompt

    async def extract(self, sampled_files: Sequence[Path]):
        try:
            def _run_generation():
                uploads = [self._ensure_upload(path) for path in sampled_files]
                response = client.models.generate_content(
                    model=self.model,
                    contents=[*uploads, self.build_prompt(sampled_files)],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        thinking_config=types.ThinkingConfig(include_thoughts=False, thinking_budget=-1),
                    ),
                )
                return _extract_text_from_response(response)

            content = await asyncio.to_thread(_run_generation)
            result = json.loads(content)

            if isinstance(result, dict) and "items" in result:
                return result["items"]
            if isinstance(result, list):
                return result
            return [result]
        except Exception as e:
            print(f"Extraction error: {e}")
            return []

    async def _process_sample(self, sampled_files: Sequence[Path]):
        if not sampled_files:
            return
        try:
            extractions = await self.extract(sampled_files)
            for item in extractions:
                source = item.get("source", "unknown")

                if item.get("type") == "statement":
                    self.add_statement(item["data"], source)
                elif item.get("type") == "argument":
                    self.add_argument(item["data"], source)
        except Exception as exc:
            print(f"Processing error: {exc}")

    def add_statement(self, data, source=None):
        artifact_obj = None
        if source and source not in self.artifacts:
            artifact_obj = {
                "id": self.next_artifact_id,
                "name": source,
                "author": "Unknown",
                "title": "Unknown",
                "year": "Unknown",
            }
            self.artifacts[source] = artifact_obj
            self.next_artifact_id += 1
        elif source:
            artifact_obj = self.artifacts[source]

        statement = {
            "id": self.next_statement_id,
            "artifact": [artifact_obj] if artifact_obj else [],
            "statement": data["statement"],
            "citations": [{"page": 1, "text": data["statement"][:100]}] if source else [],
        }

        self.statements[self.next_statement_id] = statement
        self.graph.add_node(
            self.next_statement_id,
            label=statement["statement"][:50],
            full_text=statement["statement"],
        )

        if source:
            self.statement_sources.setdefault(self.next_statement_id, set()).add(source)

        self.next_statement_id += 1
        print(f"Added Statement {statement['id']}: {statement['statement'][:100]}")
        return statement

    def add_argument(self, data, source=None):
        conclusion_id = data.get("conclusion_id")
        if conclusion_id is None or conclusion_id not in self.statements:
            return None

        premise_ids = data.get("premise_ids", [])
        for p in premise_ids:
            if p not in self.statements:
                return None

        argument = {
            "id": self.next_argument_id,
            "premise": premise_ids,
            "conclusion": conclusion_id,
            "desc": data.get("desc", ""),
        }

        self.arguments[self.next_argument_id] = argument

        if source:
            self.argument_sources.setdefault(self.next_argument_id, set()).add(source)

        for premise_id in premise_ids:
            self.graph.add_edge(
                premise_id,
                conclusion_id,
                argument_id=self.next_argument_id,
                desc=argument["desc"],
            )

        self.next_argument_id += 1
        print(f"Added Argument {argument['id']}: {argument['premise']} → {argument['conclusion']}")
        return argument

    def to_graph_data(self) -> GraphData:
        statements: List[SchemaStatement] = []
        id_to_statement: Dict[int, SchemaStatement] = {}

        for raw in self.statements.values():
            artifacts: List[SchemaArtifact] = []
            for art in raw.get("artifact", []):
                if not art:
                    continue
                artifacts.append(
                    SchemaArtifact(
                        id=art.get("id"),
                        name=art.get("name", ""),
                        author=art.get("author", ""),
                        title=art.get("title", ""),
                        year=art.get("year", ""),
                    )
                )

            citations: List[SchemaQuote] = []
            for cite in raw.get("citations", []):
                if not cite:
                    continue
                citations.append(
                    SchemaQuote(
                        page=cite.get("page", 0),
                        text=cite.get("text", ""),
                    )
                )

            stmt = SchemaStatement(
                id=raw["id"],
                artifact=artifacts,
                statement=raw["statement"],
                citations=citations,
            )
            statements.append(stmt)
            id_to_statement[stmt.id] = stmt

        arguments: List[SchemaArgument] = []
        for raw in self.arguments.values():
            try:
                premises = [id_to_statement[p] for p in raw.get("premise", [])]
                conclusion = id_to_statement[raw["conclusion"]]
            except KeyError:
                # skip malformed or out-of-sync argument entries
                continue

            arg = SchemaArgument(
                id=raw["id"],
                premise=premises,
                conclusion=conclusion,
                desc=raw.get("desc", ""),
            )
            arguments.append(arg)

        return GraphData(statements=statements, arguments=arguments)

    async def worker(self, worker_id, iterations):
        for i in range(iterations):
            print(f"Worker {worker_id} iteration {i + 1}")
            sampled_files = self._sample_files()
            await self._process_sample(sampled_files)

    @staticmethod
    def _allocate_iterations(iterations: int, workers: int) -> List[int]:
        workers = max(1, workers)
        iterations = max(0, iterations)
        base, remainder = divmod(iterations, workers)
        allocation: List[int] = []
        for idx in range(workers):
            allocation.append(base + (1 if idx < remainder else 0))
        return allocation

    async def run_async(self, iterations, workers):
        if not self.file_paths or iterations <= 0:
            return
        allocation = self._allocate_iterations(iterations, workers)
        tasks = [self.worker(i, count) for i, count in enumerate(allocation) if count > 0]
        if tasks:
            await asyncio.gather(*tasks)

    async def advance(self, iterations: int = 50, workers: int = 5) -> None:
        await self.run_async(iterations, workers)

    async def process_files(self, files: Sequence[Path], iterations: int = 1) -> None:
        if iterations <= 0:
            return
        targets = [Path(f).resolve() for f in files]
        if not targets:
            return
        for target in targets:
            self.add_source(target)
        for _ in range(iterations):
            await self._process_sample(targets)

    def save_results(self, suffix=""):
        data = {
            "statements": list(self.statements.values()),
            "arguments": list(self.arguments.values()),
            "statement_sources": {k: list(v) for k, v in self.statement_sources.items()},
            "argument_sources": {k: list(v) for k, v in self.argument_sources.items()},
        }

        filename = f"graph{suffix}.json"
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build a cross-text philosophical DAG using Gemini.")
    parser.add_argument("files", nargs="+", type=Path, help="Paths to text or PDF files to analyse")
    parser.add_argument("--sample-size", type=int, default=5, help="Number of files to sample per iteration")
    parser.add_argument("--iterations", type=int, default=20, help="Total extraction iterations to run")
    parser.add_argument("--workers", type=int, default=5, help="Parallel workers to launch")
    args = parser.parse_args()

    dag = MultiDocumentPhilosophyDAG(sample_size=args.sample_size)
    for file_path in args.files:
        dag.add_source(file_path)

    asyncio.run(dag.advance(iterations=args.iterations, workers=args.workers))
    dag.save_results()

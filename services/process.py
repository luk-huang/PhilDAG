import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from schemas.schema import GraphData
from services.explainer import MultiDocumentPhilosophyDAG

load_dotenv()

_DAG_INSTANCE: Optional[MultiDocumentPhilosophyDAG] = None


def _get_dag() -> MultiDocumentPhilosophyDAG:
    global _DAG_INSTANCE
    if _DAG_INSTANCE is None:
        sample_size = int(os.getenv("DAG_SAMPLE_SIZE", "5"))
        _DAG_INSTANCE = MultiDocumentPhilosophyDAG(sample_size=sample_size)
    return _DAG_INSTANCE


async def analyze(file_path: Path) -> GraphData:
    dag = _get_dag()
    dag.add_source(file_path)

    focus_iterations = max(1, int(os.getenv("DAG_FOCUS_ITERATIONS", "3")))
    await dag.process_files([file_path], iterations=focus_iterations)

    background_iterations = int(os.getenv("DAG_BACKGROUND_ITERATIONS", "0"))
    background_workers = int(os.getenv("DAG_BACKGROUND_WORKERS", "1"))
    if background_iterations > 0:
        await dag.advance(iterations=background_iterations, workers=background_workers)

    print(f"Updated DAG with {file_path}")
    return dag.to_graph_data()

async def extract_graph(file_path: Path) -> GraphData:
    return await analyze(file_path)


if __name__ == "__main__":
    import asyncio

    test_path = Path("text/plato_republic_514b-518d_allegory-of-the-cave.pdf")
    if not test_path.exists():
        raise FileNotFoundError(test_path)

    result = asyncio.run(analyze(test_path))
    print(result.model_dump())

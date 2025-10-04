import json
import os
from openai import OpenAI
from dotenv import load_dotenv
import networkx as nx
import asyncio
import random
import glob
from pydantic import BaseModel
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
class Quote(BaseModel):
    page: int
    text: str
class Artifact(BaseModel):
    id: int
    name: str
    author: str
    title: str
    year: str
class MultiDocumentPhilosophyDAG:
    def __init__(self, text_dir: str, sample_size: int):
        self.text_dir = text_dir
        self.sample_size = sample_size
        self.current_texts = {}  # filename -> text content for current sample
        self.all_files = []
        self.statements = {}  # id -> Statement dict
        self.arguments = {}   # id -> Argument dict
        self.artifacts = {}   # filename -> Artifact
        self.graph = nx.DiGraph()
        self.next_statement_id = 1
        self.next_argument_id = 1
        self.next_artifact_id = 1
        self.statement_sources = {}  # statement_id -> set of source files
        self.argument_sources = {}   # argument_id -> set of source files
        
        # Load all available text files
        self._load_all_files()
    
    def _load_all_files(self):
        """Load list of all text files in the directory"""
        text_pattern = os.path.join(self.text_dir, "*.txt")
        self.all_files = glob.glob(text_pattern)
        print(f"Found {len(self.all_files)} text files in {self.text_dir}")
    
    def _sample_texts(self):
        """Sample a random set of texts uniformly"""
        # Sample uniformly from all available files
        sample_files = random.sample(self.all_files, min(self.sample_size, len(self.all_files)))
        
        # Load the texts
        self.current_texts = {}
        for filepath in sample_files:
            with open(filepath, 'r', encoding='utf-8') as f:
                self.current_texts[os.path.basename(filepath)] = f.read()
    
    def build_prompt(self):
        """Build prompt with multiple texts and existing graph context"""
        prompt = """You are analyzing multiple philosophical texts to build a unified Directed Acyclic Graph (DAG) of logical arguments.
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
        # Show ALL statements (removed the limit)
        if self.statements:
            for id in sorted(self.statements.keys()):
                stmt = self.statements[id]
                sources = self.statement_sources.get(id, set())
                source_str = f" [from: {', '.join(sources)}]" if sources else ""
                prompt += f"Statement {id}: \"{stmt['statement']}\"{source_str}\n"
        else:
            prompt += "[Empty - no statements yet]\n"
        
        prompt += "\nCURRENT ARGUMENTS:\n"
        # Show ALL arguments (removed the limit)
        if self.arguments:
            for id in sorted(self.arguments.keys()):
                arg = self.arguments[id]
                prompt += f"Argument {id}: {arg['premise']} → {arg['conclusion']}: {arg['desc']}\n"
        else:
            prompt += "[Empty - no arguments yet]\n"
        
        prompt += f"\nTotal statements: {len(self.statements)}, Total arguments: {len(self.arguments)}\n"
        
        prompt += "\nPHILOSOPHICAL TEXTS TO ANALYZE:\n---\n"
        
        for filename, text in self.current_texts.items():
            prompt += f"\n[SOURCE: {filename}]\n{text}\n\n"
        
        prompt += """---
Extract 5-10 new items (statements or arguments) from these texts.
Return JSON as {"items": [...]} where each item has:
- "type": either "statement" or "argument"  
- "data": the relevant data for that type
- "source": filename(s) this item comes from
For statements:
{"type": "statement", "data": {"statement": "The claim in clear modern language"}, "source": "filename.txt"}
For arguments (can only use existing statement IDs):
{"type": "argument", "data": {"premise_ids": [1, 2], "conclusion_id": 3, "desc": "Brief description of the reasoning"}, "source": "filename.txt"}
IMPORTANT: Arguments can ONLY reference statement IDs that already exist."""
        
        return prompt
    
    async def extract(self):
        """Extract statements and arguments from current sample"""
        try:
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model="gpt-5-mini",
                messages=[{"role": "user", "content": self.build_prompt()}],
                response_format={"type": "json_object"},
                max_completion_tokens=10000,
            )
            content = response.choices[0].message.content
            result = json.loads(content)
            
            if isinstance(result, dict) and 'items' in result:
                return result['items']
            elif isinstance(result, list):
                return result
            else:
                return [result]
        except Exception as e:
            print(f"Extraction error: {e}")
            return []
    
    def add_statement(self, data, source=None):
        """Add a statement to the graph"""
        # Get or create artifact for source
        artifact_obj = None
        if source and source not in self.artifacts:
            artifact_obj = {
                'id': self.next_artifact_id,
                'name': source,
                'author': 'Unknown',
                'title': 'Unknown',
                'year': 'Unknown'
            }
            self.artifacts[source] = artifact_obj
            self.next_artifact_id += 1
        elif source:
            artifact_obj = self.artifacts[source]
        
        statement = {
            'id': self.next_statement_id,
            'artifact': [artifact_obj] if artifact_obj else [],
            'statement': data['statement'],
            'citations': [{'page': 1, 'text': data['statement'][:100]}] if source else []
        }
        
        self.statements[self.next_statement_id] = statement
        self.graph.add_node(self.next_statement_id, 
                          label=statement['statement'][:50],
                          full_text=statement['statement'])
        
        # Track source
        if source:
            if self.next_statement_id not in self.statement_sources:
                self.statement_sources[self.next_statement_id] = set()
            self.statement_sources[self.next_statement_id].add(source)
        
        self.next_statement_id += 1
        print(f"Added Statement {statement['id']}: {statement['statement'][:100]}")
        return statement
    
    def add_argument(self, data, source=None):
        """Add an argument to the graph"""
        conclusion_id = data.get('conclusion_id')
        
        if conclusion_id is None or conclusion_id not in self.statements:
            return None
        
        premise_ids = data.get('premise_ids', [])
        for p in premise_ids:
            if p not in self.statements:
                return None
        
        argument = {
            'id': self.next_argument_id,
            'premise': premise_ids,
            'conclusion': conclusion_id,
            'desc': data.get('desc', '')
        }
        
        self.arguments[self.next_argument_id] = argument
        
        # Track source
        if source:
            if self.next_argument_id not in self.argument_sources:
                self.argument_sources[self.next_argument_id] = set()
            self.argument_sources[self.next_argument_id].add(source)
        
        # Add edges to graph
        for premise_id in premise_ids:
            self.graph.add_edge(premise_id, conclusion_id, 
                              argument_id=self.next_argument_id,
                              desc=argument['desc'])
        
        self.next_argument_id += 1
        print(f"Added Argument {argument['id']}: {argument['premise']} → {argument['conclusion']}")
        return argument
    
    async def worker(self, worker_id, iterations):
        """Worker function that processes iterations"""
        for i in range(iterations):
            print(f"Worker {worker_id} iteration {i+1}")
            # Sample new texts for this iteration
            self._sample_texts()
            
            if not self.current_texts:
                continue
            
            try:
                extractions = await self.extract()
                for item in extractions:
                    source = item.get('source', 'unknown')
                    
                    if item['type'] == 'statement':
                        stmt = self.add_statement(item['data'], source)
                    
                    elif item['type'] == 'argument':
                        arg = self.add_argument(item['data'], source)
            
            except Exception as e:
                pass
    
    async def run_async(self, iterations, workers):
        """Run multiple workers in parallel"""
        iters_per_worker = iterations // workers
        tasks = [self.worker(i, iters_per_worker) for i in range(workers)]
        await asyncio.gather(*tasks)
    
    def run(self, iterations: int = 100, workers: int = 5):
        """Main entry point for running the extraction"""
        print(f"Running {iterations} iterations with {workers} parallel workers...")
        
        asyncio.run(self.run_async(iterations, workers))
        
        self.save_results()
    
    def save_results(self, suffix=""):
        """Save the graph data and report"""
        data = {
            'statements': list(self.statements.values()),
            'arguments': list(self.arguments.values()),
            'statement_sources': {k: list(v) for k, v in self.statement_sources.items()},
            'argument_sources': {k: list(v) for k, v in self.argument_sources.items()},
        }
        
        filename = f'graph{suffix}.json'
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
if __name__ == "__main__":
    dag = MultiDocumentPhilosophyDAG("texts/processed/", sample_size=5)
    # dag.run(iterations=4, workers=2)
    dag.run(iterations=100, workers=20)
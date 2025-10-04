import json
import os
from openai import OpenAI
from dotenv import load_dotenv
import networkx as nx
import asyncio

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class PhilosophyDAG:
    def __init__(self, text_file):
        with open(text_file, 'r') as f:
            self.text = f.read()
        self.nodes = {}
        self.graph = nx.DiGraph()
        self.next_id = 1
    
    def build_prompt(self):
        prompt = """You are analyzing a philosophical text to build a Directed Acyclic Graph (DAG) of logical arguments.

TASK: Extract ONE STATEMENT from the text below.

KEY CONCEPT:
- A statement is any claim or assertion from the text
- It can have 0, 1, or multiple supporting statements
- An "axiom" is simply a node with no incoming edges (no supporting statements)
- Every statement becomes a node in our DAG
- Build up from foundational claims - don't jump to complex conclusions without the supporting structure

IMPORTANT RULES:
1. Extract ONE statement/claim from the text
2. Supporting statements must ONLY reference existing nodes (by their exact text) or be empty
3. Be precise and clear - the text may use old or archaic English, so feel free to modernize the language
4. Don't repeat existing nodes
5. Each iteration creates exactly ONE new node
6. Only extract a claim if it can be FULLY justified by existing nodes, otherwise extract simpler/foundational claims first
7. Take your time - you have hundreds of iterations to complete this task, so each iteration should do a VERY SMALL part of the work
8. Every axiom should be very small and precise - atomic facts that cannot be broken down further
9. Every argument (statement that follows from other nodes) must be very well justified, clear, and precise

CURRENT NODES IN THE DAG:
"""
        if self.nodes:
            for id, node in self.nodes.items():
                prompt += f"Node {id}: \"{node['statement']}\"\n"
        else:
            prompt += "[Empty - this will be the first node]\n"
        
        prompt += f"""
PHILOSOPHICAL TEXT TO ANALYZE:
---
{self.text}
---

Remember: You have hundreds of iterations to work through this text. Don't rush. Extract small, precise, atomic pieces of reasoning. Build the logical structure gradually from the ground up.

Extract ONE new statement that hasn't been identified yet.

Return JSON in this exact format:
{{
  "statement": "The claim/statement from the text (in clear modern language)",
  "supporting_statements": []  // Empty array if no support (axiom), or ["support1", "support2", ...] if supported
}}

Example with no support (axiom): {{"statement": "The servant does not know geometry", "supporting_statements": []}}
Example with support: {{"statement": "The soul is immortal", "supporting_statements": ["The servant knows without being taught", "Knowledge comes from before birth"]}}

You MUST return valid JSON format."""
        return prompt
    
    async def extract(self):
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model="gpt-5-mini",
            messages=[{"role": "user", "content": self.build_prompt()}],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    
    def add_node(self, extraction):
        # Check if statement already exists
        for node in self.nodes.values():
            if node['statement'].lower() == extraction['statement'].lower():
                return None  # Skip duplicate
        
        node = {
            'id': self.next_id,
            'statement': extraction['statement'],
            'supports': []
        }
        
        for stmt in extraction.get('supporting_statements', []):
            # Only link to existing nodes, don't create new ones
            for id, n in self.nodes.items():
                if n['statement'].lower() == stmt.lower():
                    node['supports'].append(id)
                    self.graph.add_edge(id, node['id'])
                    break
        
        self.nodes[node['id']] = node
        self.graph.add_node(node['id'])
        self.next_id += 1
        return node
    
    async def worker(self, worker_id, iterations):
        for i in range(iterations):
            try:
                extraction = await self.extract()
                node = self.add_node(extraction)
                if node:
                    print(f"\n[W{worker_id}:{i+1:3d}] Added: {node['statement'][:300]}")
            except Exception as e:
                print(f"\n[W{worker_id}:{i+1:3d}] Error: {e}")
    
    async def run_async(self, iterations, workers):
        iters_per_worker = iterations // workers
        tasks = [self.worker(i, iters_per_worker) for i in range(workers)]
        await asyncio.gather(*tasks)
    
    def run(self, iterations, workers):
        print(f"Running {iterations} iterations with {workers} parallel workers...")
        asyncio.run(self.run_async(iterations, workers))
        print(f"\n\nCompleted {iterations} iterations. Total nodes: {len(self.nodes)}")
        self.save_results()
    
    def save_results(self):
        with open('graph.json', 'w') as f:
            json.dump(self.nodes, f, indent=2)
        
        with open('report.txt', 'w') as f:
            f.write(f"EXTRACTED {len(self.nodes)} NODES\n\n")
            f.write("AXIOMS (no supporting statements):\n")
            for n in self.nodes.values():
                if not n['supports']:
                    f.write(f"- {n['statement']}\n")
            f.write("\nSUPPORTED CONCLUSIONS:\n")
            for n in self.nodes.values():
                if n['supports']:
                    f.write(f"- {n['statement']}\n")
                    for sid in n['supports']:
                        f.write(f"  ← {self.nodes[sid]['statement']}\n")
        
        print(f"Saved to graph.json and report.txt")

if __name__ == "__main__":
    dag = PhilosophyDAG("jason/famine_singer.txt")
    dag.run(200, 10)
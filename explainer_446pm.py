import json
import os
from openai import OpenAI
from dotenv import load_dotenv
import networkx as nx
import asyncio
from schemas.analysis import Quote, Artifact, Statement, Argument

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class PhilosophyDAG:
    def __init__(self, text_file):
        with open(text_file, 'r') as f:
            self.text = f.read()
        self.statements = {}  # id -> Statement dict
        self.arguments = {}   # id -> Argument dict
        self.graph = nx.DiGraph()
        self.next_statement_id = 1
        self.next_argument_id = 1
    
    def build_prompt(self):
        prompt = """You are analyzing a philosophical text to build a Directed Acyclic Graph (DAG) of logical arguments.

TASK: Extract MULTIPLE items from the text below - either statements (claims) or arguments (justifications).

KEY CONCEPTS:
- A STATEMENT is any claim or assertion from the text (unsubstantiated on its own)
- An ARGUMENT connects premise statements to a conclusion statement with justification
- Statements are the nodes; Arguments are the edges that connect them
- Build up from foundational claims to complex conclusions

IMPORTANT RULES:
1. Extract 1-5 items per response (statements or arguments)
2. Statements should be atomic, clear claims
3. Arguments can ONLY reference existing statement IDs as premises and conclusion
4. Use clear, modern language (translate archaic English if needed)
5. Statements and arguments are immutable once created
6. Take your time building the logical structure gradually

CURRENT STATEMENTS IN THE DAG:
"""
        if self.statements:
            for id, stmt in self.statements.items():
                prompt += f"Statement {id}: \"{stmt['statement']}\"\n"
        else:
            prompt += "[Empty - no statements yet]\n"
        
        prompt += "\nCURRENT ARGUMENTS:\n"
        if self.arguments:
            for id, arg in self.arguments.items():
                prompt += f"Argument {id}: {arg['premise']} → {arg['conclusion']}: {arg['desc']}\n"
        else:
            prompt += "[Empty - no arguments yet]\n"
        
        prompt += f"""

PHILOSOPHICAL TEXT TO ANALYZE:
---
{self.text}
---

Extract 1-5 new items (statements or arguments) that haven't been identified yet.

Return JSON as a list where each element has:
- "type": either "statement" or "argument"  
- "data": the relevant data for that type

For statements:
{{"type": "statement", "data": {{"statement": "The claim in clear modern language"}}}}

For arguments (can only use existing statement IDs):
{{"type": "argument", "data": {{"premise_ids": [1, 2], "conclusion_id": 3, "desc": "Brief description of the reasoning"}}}}

IMPORTANT: Arguments can ONLY reference statement IDs that already exist. You cannot create new conclusions inline.

Return valid JSON array format."""
        return prompt
    
    async def extract(self):
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model="gpt-5-mini",
            messages=[{"role": "user", "content": self.build_prompt()}],
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        # The model returns a JSON object, we need to extract the array
        result = json.loads(content)
        if isinstance(result, dict) and 'items' in result:
            return result['items']
        elif isinstance(result, list):
            return result
        else:
            # Wrap single item in list
            return [result]
    
    def add_statement(self, data):
        # Create statement dict matching the Statement schema structure
        # Statement schema has: id, artifact (list), statement (str), citations (list)
        statement = {
            'id': self.next_statement_id,
            'artifact': [],  # Empty for now as requested
            'statement': data['statement'],
            'citations': []  # Empty for now as requested
        }
        
        self.statements[self.next_statement_id] = statement
        self.graph.add_node(self.next_statement_id, label=statement['statement'][:50])
        self.next_statement_id += 1
        return statement
    
    def add_argument(self, data):
        # Arguments are immutable - they can only reference existing statements
        conclusion_id = data.get('conclusion_id')
        
        # Validate conclusion exists
        if conclusion_id is None or conclusion_id not in self.statements:
            return None  # Can't create argument without valid conclusion
        
        # Validate all premise IDs exist
        premise_ids = data.get('premise_ids', [])
        for p in premise_ids:
            if p not in self.statements:
                return None  # All premises must exist
        
        # Create argument dict matching the Argument schema structure
        # Argument schema has: id, premise (list of Statements), conclusion (Statement), desc
        # For efficiency, we store IDs and dereference when needed
        argument = {
            'id': self.next_argument_id,
            'premise': premise_ids,  # List of Statement IDs (not full Statement objects for now)
            'conclusion': conclusion_id,  # Statement ID (not full Statement object for now)
            'desc': data.get('desc', '')
        }
        
        self.arguments[self.next_argument_id] = argument
        
        # Add edges to graph
        for premise_id in premise_ids:
            self.graph.add_edge(premise_id, conclusion_id, 
                               argument_id=self.next_argument_id,
                               desc=argument['desc'])
        
        self.next_argument_id += 1
        return argument
    
    async def worker(self, worker_id, iterations):
        for i in range(iterations):
            try:
                extractions = await self.extract()
                for item in extractions:
                    if item['type'] == 'statement':
                        stmt = self.add_statement(item['data'])
                        if stmt:
                            print(f"\n[W{worker_id}:{i+1:3d}] Added Statement {stmt['id']}: {stmt['statement'][:100]}")
                    elif item['type'] == 'argument':
                        arg = self.add_argument(item['data'])
                        if arg:
                            print(f"\n[W{worker_id}:{i+1:3d}] Added Argument {arg['id']}: {arg['premise']} → {arg['conclusion']}")
            except Exception as e:
                print(f"\n[W{worker_id}:{i+1:3d}] Error: {e}")
    
    async def run_async(self, iterations, workers):
        iters_per_worker = iterations // workers
        tasks = [self.worker(i, iters_per_worker) for i in range(workers)]
        await asyncio.gather(*tasks)
    
    def run(self, iterations, workers):
        print(f"Running {iterations} iterations with {workers} parallel workers...")
        asyncio.run(self.run_async(iterations, workers))
        print(f"\n\nCompleted {iterations} iterations.")
        print(f"Total statements: {len(self.statements)}")
        print(f"Total arguments: {len(self.arguments)}")
        self.save_results()
    
    def save_results(self):
        # Save raw data
        data = {
            'statements': self.statements,
            'arguments': self.arguments
        }
        with open('graph.json', 'w') as f:
            json.dump(data, f, indent=2)
        
        # Generate report
        with open('report.txt', 'w') as f:
            f.write(f"EXTRACTED {len(self.statements)} STATEMENTS and {len(self.arguments)} ARGUMENTS\n\n")
            
            # Find axioms (statements with no incoming arguments)
            axioms = []
            conclusions = set()
            for arg in self.arguments.values():
                conclusions.add(arg['conclusion'])
            
            for stmt_id, stmt in self.statements.items():
                if stmt_id not in conclusions:
                    axioms.append(stmt)
            
            f.write("AXIOMS (statements with no supporting arguments):\n")
            for stmt in axioms:
                f.write(f"[{stmt['id']}] {stmt['statement']}\n")
            
            f.write("\n\nSTATEMENTS WITH ARGUMENTS:\n")
            for stmt_id in conclusions:
                stmt = self.statements[stmt_id]
                f.write(f"\n[{stmt_id}] {stmt['statement']}\n")
                
                # Find arguments that conclude this statement
                for arg in self.arguments.values():
                    if arg['conclusion'] == stmt_id:
                        f.write(f"  ← Argument {arg['id']}: {arg['desc']}\n")
                        for premise_id in arg['premise']:
                            f.write(f"    • [{premise_id}] {self.statements[premise_id]['statement']}\n")
        
        print(f"Saved to graph.json and report.txt")

if __name__ == "__main__":
    dag = PhilosophyDAG("text/famine_singer.txt")
    dag.run(100, 10)
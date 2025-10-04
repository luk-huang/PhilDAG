#####
# Representation for the DAG
#####

class Edge:
    def __init__(self, from_node_list, to_node, tag = "Logic"):
        self.from_node_list = from_node_list
        self.to_node = to_node
        self.tag = tag

class Node:
    def __init__(self, id, text, node_type = "Statement"):
        self.id = id
        self.text = text
        self.node_type = node_type 
        self.clauses = []

    def add_clause(self, edge: Edge):
        self.clauses.append(edge)

class DAG:
    def __init__(self, n, text_list: list[str]) :
        self.dag = []
        for i, text in enumerate(text_list):
            new_node = Node(i, text)
            self.dag.append(new_node)
        self.n = n
        self.text_list = text_list

    def get_sources(self):
        sources = []
        for node in self.dag:
            if len(node.clauses) == 0:
                sources.append(node)
        return sources
    
    def add_edge(self, edge: Edge):
        self.dag[edge.to_node].add_clause(edge)
    
    def get_derived_conclusions(self, true_list: list[bool], tag = []):
        derived_conclusions = []
        for i, node in enumerate(self.dag):
            if true_list[i]:
                derived_conclusions.append(i)
            else:
                some_clause_true = False
                for edge in node.clauses:
                    clause_true = True
                    if (edge.tag not in tag) and (len(tag) > 0):
                        continue
                    for subnode in edge.from_node_list:
                        if (not true_list[subnode.id]) and (subnode.id not in derived_conclusions):
                            clause_true = False
                            break
                    if clause_true:
                        some_clause_true = True
                        break
                if some_clause_true:
                    derived_conclusions.append(i)
        return derived_conclusions
    

    


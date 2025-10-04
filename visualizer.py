import json

# Load the graph data
with open('graph.json', 'r') as f:
    graph_data = json.load(f)

statements = graph_data.get('statements', [])
arguments = graph_data.get('arguments', [])

nodes = []
edges = []

# Add statement nodes (circles, blue/green based on whether they have supporting arguments)
for stmt in statements:
    stmt_id = stmt['id']
    # Check if this statement is a conclusion of any argument
    is_conclusion = any(arg['conclusion'] == stmt_id for arg in arguments)
    
    color = '#90EE90' if is_conclusion else '#87CEEB'  # Green if supported, blue if axiom
    
    nodes.append({
        'id': f'stmt_{stmt_id}',
        'label': stmt['statement'][:40] + '...' if len(stmt['statement']) > 40 else stmt['statement'],
        'title': f"Statement {stmt_id}: {stmt['statement']}",
        'color': color,
        'shape': 'dot',
        'size': 25,
        'group': 'statement'
    })

# Add argument nodes (squares, orange)
for arg in arguments:
    arg_id = arg['id']
    nodes.append({
        'id': f'arg_{arg_id}',
        'label': f"Arg {arg_id}",
        'title': f"Argument {arg_id}: {arg['desc']}",
        'color': '#FFB366',  # Orange for arguments
        'shape': 'square',
        'size': 15,
        'group': 'argument'
    })
    
    # Add edges from premises to this argument
    for premise_id in arg['premise']:
        edges.append({
            'from': f'stmt_{premise_id}',
            'to': f'arg_{arg_id}',
            'arrows': 'to',
            'color': {'color': '#848484'},
            'dashes': False
        })
    
    # Add edge from this argument to its conclusion
    edges.append({
        'from': f'arg_{arg_id}',
        'to': f"stmt_{arg['conclusion']}",
        'arrows': 'to',
        'color': {'color': '#FF6B6B'},  # Red for conclusion edges
        'width': 3,
        'dashes': False
    })

# Generate HTML
html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Philosophy Bipartite DAG</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.js"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.css" rel="stylesheet">
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        #graph {{
            height: 800px;
            border: 2px solid #ddd;
            background-color: white;
            border-radius: 8px;
        }}
        .legend {{
            margin: 20px 0;
            padding: 15px;
            background-color: white;
            border-radius: 8px;
            border: 1px solid #ddd;
        }}
        .legend h3 {{
            margin-top: 0;
        }}
        .legend-item {{
            display: inline-block;
            margin-right: 30px;
            margin-bottom: 10px;
        }}
        .legend-symbol {{
            display: inline-block;
            width: 20px;
            height: 20px;
            margin-right: 8px;
            vertical-align: middle;
            border: 2px solid #333;
        }}
        .circle {{
            border-radius: 50%;
        }}
        .square {{
            border-radius: 2px;
        }}
        .axiom {{
            background-color: #87CEEB;
        }}
        .supported {{
            background-color: #90EE90;
        }}
        .argument-node {{
            background-color: #FFB366;
        }}
        .info {{
            margin: 20px 0;
            padding: 15px;
            background-color: #e8f4ff;
            border-radius: 8px;
            border: 1px solid #b3d9ff;
        }}
    </style>
</head>
<body>
    <h1>Philosophy Argument DAG Visualization</h1>
    
    <div class="info">
        <strong>How to read this graph:</strong>
        <ul>
            <li>Statements (claims) flow into Arguments (reasoning)</li>
            <li>Arguments flow into the Statements they support (conclusions)</li>
            <li>Gray arrows: premises supporting an argument</li>
            <li>Red arrows: argument supporting its conclusion</li>
        </ul>
    </div>
    
    <div class="legend">
        <h3>Legend</h3>
        <div class="legend-item">
            <span class="legend-symbol circle axiom"></span>
            <span>Axiom Statement (no supporting arguments)</span>
        </div>
        <div class="legend-item">
            <span class="legend-symbol circle supported"></span>
            <span>Supported Statement (has arguments)</span>
        </div>
        <div class="legend-item">
            <span class="legend-symbol square argument-node"></span>
            <span>Argument (connects premises to conclusion)</span>
        </div>
    </div>
    
    <div id="graph"></div>
    
    <script>
        var nodes = new vis.DataSet({json.dumps(nodes)});
        var edges = new vis.DataSet({json.dumps(edges)});
        
        var container = document.getElementById('graph');
        var data = {{
            nodes: nodes,
            edges: edges
        }};
        
        var options = {{
            physics: {{
                barnesHut: {{
                    gravitationalConstant: -4000,
                    centralGravity: 0.3,
                    springLength: 150,
                    springConstant: 0.04,
                    damping: 0.09,
                    avoidOverlap: 0.5
                }},
                stabilization: {{
                    iterations: 200,
                    updateInterval: 50
                }}
            }},
            edges: {{
                smooth: {{
                    type: 'cubicBezier',
                    forceDirection: 'none'
                }},
                width: 2
            }},
            nodes: {{
                font: {{
                    size: 14,
                    face: 'Arial'
                }},
                borderWidth: 2,
                shadow: true,
                borderWidthSelected: 3
            }},
            interaction: {{
                hover: true,
                tooltipDelay: 200,
                zoomView: true,
                dragView: true
            }},
            layout: {{
                hierarchical: {{
                    enabled: false  // Set to true if you want a hierarchical layout
                }}
            }}
        }};
        
        var network = new vis.Network(container, data, options);
        
        // Add click event to show full details
        network.on("click", function(params) {{
            if (params.nodes.length > 0) {{
                var nodeId = params.nodes[0];
                var node = nodes.get(nodeId);
                if (node) {{
                    console.log("Clicked:", node.title);
                }}
            }}
        }});
    </script>
    
    <div style="margin-top: 20px; padding: 15px; background-color: #f0f0f0; border-radius: 8px;">
        <strong>Statistics:</strong>
        Total Statements: {len(statements)} | 
        Total Arguments: {len(arguments)} | 
        Axioms: {sum(1 for stmt in statements if not any(arg['conclusion'] == stmt['id'] for arg in arguments))}
    </div>
</body>
</html>
"""

# Write the HTML file
with open('philosophy_dag.html', 'w') as f:
    f.write(html_content)

print(f"Created philosophy_dag.html - open it in your browser")
print(f"\nGraph Statistics:")
print(f"  - {len(statements)} statements")
print(f"  - {len(arguments)} arguments") 
print(f"  - {sum(1 for stmt in statements if not any(arg['conclusion'] == stmt['id'] for arg in arguments))} axioms (unsupported statements)")
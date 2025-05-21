import pandas as pd
import os
from collections import defaultdict
from pyvis.network import Network

# === CSV file list ===
csv_files = [
    'cyberbullying_tweets.csv',
    'device.csv',
    'device_cleaned.csv',
    'email_2010_part1.csv',
    'email_2010_part2.csv',
    'email_2011.csv',
    'file_cleaned.csv',
    'logon_clean.csv',
    'http_cleaned.csv'
]

# === Field → Entity Type mapping ===
field_type_map = {
    'user': 'user',
    'pc': 'pc',
    'to': 'email',
    'cc': 'email',
    'bcc': 'email',
    'from': 'email',
    'filename': 'file',
    'file': 'file',
    'tweet': 'tweet',
    'content': 'content',
    'attachment': 'file',
    'activity': 'activity',
    'activity_binary': 'activity',
    'activity_binary_hour': 'activity',
    'hour': 'time',
    'dayofweek': 'time',
    'date': 'date',
    'date/time': 'date',
    'content_clean': 'content',
    'id': 'log',
    'size': 'value',
    'year': 'time',
    'url': 'url'
}

# === Node color by entity type ===
type_colors = {
    'user': '	#a7d3f2', #Soft Sky Blue
    'pc': '	#b6e2d3', #Light Mint Green
    'email': '#d7bde2', #Pale Lavender
    'file': '#f7b7a3', #Light Coral Pink
    'tweet': '#f9d5a7', #Pastel Orange
    'activity': '#faf3a0', #Lemon Cream Yellow
    'content': '#b2ebf2', #Pale Turquoise
    'time': '#fadadd', #Baby Pink
    'date': '	#ffe5b4', #Peach Puff
    'log': '#d4eac8', #Pastel Green
    'value': '#f5e1a4', #Cream Beige
    'other': '#c9daea',#Misty Blue
    'url': '#e3d0ff' #Powder Lilac
}

# === Settings ===
MAX_RECORDS = 60
MAX_EDGES = 2000
skip_fields = {'content', 'content_clean', 'attachment'}

nodes = dict()
edges = set()

# === Process each CSV file ===
for file_path in csv_files:
    if not os.path.exists(file_path):
        print(f"❌ File not found：{file_path}")
        continue

    df = pd.read_csv(file_path)
    df = df.head(MAX_RECORDS)

    for _, row in df.iterrows():
        field_nodes = []
        tooltips = []

        for col in row.index:
            if col.lower() in skip_fields:
                continue

            val = str(row[col]).strip()
            if pd.isna(val) or len(val) < 2 or val.lower() == 'nan':
                continue

            field_type = field_type_map.get(col.lower(), 'other')
            node_id = f"{field_type}:{val}"

            # Create node if not already exists
            if node_id not in nodes:
                nodes[node_id] = {
                    'id': node_id,
                    'label': val[:18],
                    'title': f"<b>Type:</b> {field_type}<br><b>Field:</b> {col}<br><b>Value:</b> {val}",
                    'color': type_colors.get(field_type, '#dddddd')
                }

            field_nodes.append(node_id)

        # Connect all field values within the same row
        for i in range(len(field_nodes)):
            for j in range(i + 1, len(field_nodes)):
                if len(edges) >= MAX_EDGES:
                    break
                a, b = field_nodes[i], field_nodes[j]
                edges.add((a, b))

# === Build network ===
net = Network(height="900px", width="100%", bgcolor="#ffffff", font_color="#222", notebook=False)
net.force_atlas_2based(gravity=-60, central_gravity=0.01, spring_length=120, spring_strength=0.08, damping=0.4)

# === Add nodes ===
for node in nodes.values():
    net.add_node(
        node["id"],
        label=node["label"],
        title=node["title"],
        color=node["color"],
        size=14
    )

# === Add edges ===
for src, tgt in edges:
    net.add_edge(src, tgt, width=0.6)

# === Customize appearance ===
net.set_options("""
var options = {
  "nodes": {
    "font": {
      "size": 14
    },
    "borderWidth": 0
  },
  "edges": {
    "color": {
      "inherit": true
    },
    "smooth": {
      "type": "continuous"
    }
  },
  "interaction": {
    "hover": true,
    "tooltipDelay": 100
  },
  "physics": {
    "forceAtlas2Based": {
      "gravitationalConstant": -60,
      "centralGravity": 0.01,
      "springLength": 120,
      "springConstant": 0.08,
      "damping": 0.4,
      "avoidOverlap": 1
    },
    "minVelocity": 0.75,
    "solver": "forceAtlas2Based"
  }
}
""")

# === Output final HTML file ===
net.show("data_graph.html")
print(f"✅ Graph generated: data_graph.html | Nodes: {len(nodes)} | Edges: {len(edges)}")
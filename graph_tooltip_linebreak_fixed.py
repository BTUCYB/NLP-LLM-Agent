
import pandas as pd
import os
from collections import defaultdict
from pyvis.network import Network

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

type_colors = {
    'user': '#a7d3f2',
    'pc': '#b6e2d3',
    'email': '#d7bde2',
    'file': '#f7b7a3',
    'tweet': '#f9d5a7',
    'activity': '#faf3a0',
    'content': '#b2ebf2',
    'time': '#fadadd',
    'date': '#ffe5b4',
    'log': '#d4eac8',
    'value': '#f5e1a4',
    'other': '#c9daea',
    'url': '#e3d0ff'
}

MAX_RECORDS = 60
MAX_EDGES = 2000
skip_fields = {'content', 'content_clean', 'attachment'}

nodes = dict()
edges = set()

for file_path in csv_files:
    if not os.path.exists(file_path):
        continue

    df = pd.read_csv(file_path)
    if 'bcc' in df.columns:
        df['bcc'] = df['bcc'].apply(lambda x: None if str(x).strip() == '[]' else x)
    df = df.head(MAX_RECORDS)

    for _, row in df.iterrows():
        field_nodes = []

        for col in row.index:
            if col.lower() in skip_fields:
                continue

            val = str(row[col]).strip()
            if pd.isna(val) or len(val) < 2 or val.lower() == 'nan':
                continue

            field_type = field_type_map.get(col.lower(), 'other')
            node_id = f"{field_type}:{val}"

            if node_id not in nodes:
                nodes[node_id] = {
                    'id': node_id,
                    'label': val[:18],
                    'color': type_colors.get(field_type, '#dddddd'),
                    'value': val,
                    'field': col,
                    'type': field_type,
                    'file': file_path
                }

            field_nodes.append(node_id)

        for i in range(len(field_nodes)):
            for j in range(i + 1, len(field_nodes)):
                if len(edges) >= MAX_EDGES:
                    break
                edges.add((field_nodes[i], field_nodes[j]))

def compute_score(val1, val2):
    try:
        return abs(float(val1) - float(val2))
    except:
        return len(set(str(val1)).intersection(set(str(val2))))

for node_data in nodes.values():
    node_data['score'] = 0

for src, tgt in edges:
    val1 = nodes[src]['value']
    val2 = nodes[tgt]['value']
    score = compute_score(val1, val2)
    nodes[src]['score'] += score
    nodes[tgt]['score'] += score

for node_data in nodes.values():
    tooltip_lines = [
        f'score: "{node_data["score"]}"',
        f'file: "{node_data["file"]}"',
        f'type: "{node_data["type"]}"',
        f'field: "{node_data["field"]}"',
        f'value: "{node_data["value"]}"'
    ]
    node_data["title"] = "<div style='white-space: pre-line'>" + "\n".join(tooltip_lines) + "</div>"

net = Network(height="900px", width="100%", bgcolor="#ffffff", font_color="#222", notebook=False)
net.force_atlas_2based(gravity=-60, central_gravity=0.01, spring_length=120, spring_strength=0.08, damping=0.4)

for node in nodes.values():
    net.add_node(
        node["id"],
        label=node["label"],
        title=node["title"],
        color=node["color"],
        size=14
    )

for src, tgt in edges:
    net.add_edge(src, tgt, width=0.6)

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

net.show("data_graph13.html")
print(f"✅ Graph generated: data_graph13.html | Nodes: {len(nodes)} | Edges: {len(edges)}")

"""Presentation-ready end-to-end KG demo.

Usage:
    python run_pipeline.py --input raw_data.txt --max-chunks 4

The script uses AICredits when AICREDITS_API_KEY is configured.  Set
LLM_PROVIDER=ollama to use the locally installed Ollama model instead.
Neo4j loading is optional and can be enabled with --neo4j.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List

import networkx as nx
from dotenv import load_dotenv

from kg_extractor import StatefulKGExtractor, Triple
from llm_client import LLMClient, extract_json

load_dotenv()


def ontology_prompt(text: str) -> str:
    return f"""Analyze this document and create a compact knowledge-graph ontology.
Return ONLY a JSON object with exactly these keys:
{{"classes": ["..."], "properties": {{"Class": ["..."]}}, "relationships": [{{"subject": "Class", "predicate": "RELATION", "object": "Class"}}]}}
Do not invent facts that are not suggested by the document.

DOCUMENT:
{text}
"""


def extraction_prompt(text: str, ontology: Dict, seen_entities: List[str]) -> str:
    return f"""Extract factual knowledge-graph triples from the document chunk below.
Return ONLY a JSON object with this exact shape:
{{"triples": [{{"subject": "...", "predicate": "...", "object": "...", "confidence": 0.0}}]}}

Rules:
- Extract only facts supported by the text.
- Use short, consistent entity names.
- Use uppercase snake-case predicates.
- Confidence must be between 0 and 1.
- Include up to 15 high-quality triples.
- Previously seen names: {json.dumps(seen_entities[:30])}
- Ontology guidance: {json.dumps(ontology, ensure_ascii=False)}

CHUNK:
{text}
"""


def parse_triples(response: str, chunk_id: int) -> List[Triple]:
    data = extract_json(response)
    if isinstance(data, dict):
        data = data.get("triples", [])
    if not isinstance(data, list):
        return []

    triples = []
    for item in data:
        if not isinstance(item, dict):
            continue
        subject = str(item.get("subject", "")).strip()
        predicate = re.sub(r"[^A-Za-z0-9]+", "_", str(item.get("predicate", ""))).strip("_").upper()
        obj = str(item.get("object", "")).strip()
        if not subject or not predicate or not obj:
            continue
        try:
            confidence = max(0.0, min(1.0, float(item.get("confidence", 0.8))))
        except (TypeError, ValueError):
            confidence = 0.8
        triples.append(Triple(subject, predicate, obj, chunk_id, confidence))
    return triples


def node_key(name: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return key or "entity"


def build_graph(triples: List[Triple]) -> Dict:
    nodes: Dict[str, Dict] = {}
    edges: Dict[tuple, Dict] = {}
    for triple in triples:
        source_id = node_key(triple.subject)
        target_id = node_key(triple.object)
        nodes.setdefault(source_id, {"id": source_id, "name": triple.subject, "type": "Entity"})
        nodes.setdefault(target_id, {"id": target_id, "name": triple.object, "type": "Entity"})
        edge_key = (source_id, triple.predicate, target_id)
        previous = edges.get(edge_key)
        edge = {
            "source": source_id,
            "target": target_id,
            "predicate": triple.predicate,
            "confidence": triple.confidence,
            "source_chunk": triple.source_chunk,
        }
        if previous is None or triple.confidence > previous["confidence"]:
            edges[edge_key] = edge
    return {"nodes": list(nodes.values()), "edges": list(edges.values())}


def write_graphml(graph: Dict, filepath: Path):
    nx_graph = nx.MultiDiGraph()
    for node in graph["nodes"]:
        nx_graph.add_node(node["id"], name=node["name"], type=node["type"])
    for edge_index, edge in enumerate(graph["edges"]):
        nx_graph.add_edge(
            edge["source"], edge["target"], key=edge_index,
            label=edge["predicate"], confidence=edge["confidence"],
            source_chunk=edge["source_chunk"],
        )
    nx.write_graphml(nx_graph, filepath)


def write_html(graph: Dict, filepath: Path):
    nodes = [
        {"id": node["id"], "label": node["name"], "title": node["type"], "shape": "dot"}
        for node in graph["nodes"]
    ]
    edges = [
        {"from": edge["source"], "to": edge["target"], "label": edge["predicate"], "arrows": "to"}
        for edge in graph["edges"]
    ]
    content = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>LLM Knowledge Graph</title>
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<style>body{{margin:0;font-family:Arial;background:#151a21;color:#fff}}#network{{height:92vh}}#info{{padding:10px;background:#222a35}}</style>
</head><body><div id="info">Click a node to inspect it.</div><div id="network"></div>
<script>
const nodes = new vis.DataSet({json.dumps(nodes, ensure_ascii=False)});
const edges = new vis.DataSet({json.dumps(edges, ensure_ascii=False)});
const network = new vis.Network(document.getElementById('network'), {{nodes, edges}}, {{nodes:{{font:{{color:'#fff'}}}},edges:{{font:{{color:'#ddd',align:'middle'}},smooth:true}},physics:{{stabilization:true}}}});
network.on('click', p => {{ if (p.nodes.length) {{ const n=nodes.get(p.nodes[0]); document.getElementById('info').textContent='Entity: '+n.label+' | Type: '+(n.title||'Entity'); }} }});
</script></body></html>"""
    filepath.write_text(content, encoding="utf-8")


def load_into_neo4j(graph: Dict):
    from neo4j import GraphDatabase

    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "")
    if not password:
        raise RuntimeError("NEO4J_PASSWORD is required when --neo4j is used")

    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session() as session:
            session.run("CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE")
            session.run("""
                UNWIND $nodes AS item
                MERGE (e:Entity {id: item.id})
                SET e.name = item.name, e.type = item.type
            """, nodes=graph["nodes"])
            session.run("""
                UNWIND $edges AS item
                MATCH (source:Entity {id: item.source})
                MATCH (target:Entity {id: item.target})
                MERGE (source)-[r:RELATION {predicate: item.predicate}]->(target)
                SET r.confidence = item.confidence, r.source_chunk = item.source_chunk
            """, edges=graph["edges"])
    finally:
        driver.close()


def main():
    parser = argparse.ArgumentParser(description="Build a presentation-ready knowledge graph")
    parser.add_argument("--input", default="raw_data.txt")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--max-chunks", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=180)
    parser.add_argument("--overlap", type=int, default=30)
    parser.add_argument("--neo4j", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    text = input_path.read_text(encoding="utf-8")

    client = LLMClient.from_env()
    if not client.is_ready():
        raise SystemExit(
            f"LLM provider '{client.provider}' is not ready. "
            "Set AICREDITS_API_KEY or start Ollama and pull the configured model."
        )
    print(f"Using {client.provider}: {client.model}")

    extractor = StatefulKGExtractor()
    chunks = extractor.chunk_document(text, args.chunk_size, args.overlap)[: args.max_chunks]
    sample = "\n\n".join(chunk["text"] for chunk in chunks)
    ontology = extract_json(client.chat([
        {"role": "system", "content": "You are an ontology engineer. Return valid JSON only."},
        {"role": "user", "content": ontology_prompt(sample)},
    ], temperature=0.0, max_tokens=900, json_mode=True))
    if not isinstance(ontology, dict):
        ontology = {"classes": [], "properties": {}, "relationships": []}

    seen_entities: List[str] = []
    for chunk in chunks:
        try:
            response = client.chat([
                {"role": "system", "content": "You extract precise knowledge graph triples. Return valid JSON only."},
                {"role": "user", "content": extraction_prompt(chunk["text"], ontology, seen_entities)},
            ], temperature=0.1, max_tokens=900, json_mode=True)
            triples = parse_triples(response, chunk["id"])
        except (RuntimeError, ValueError) as exc:
            print(f"Chunk {chunk['id']} skipped: {exc}")
            triples = []
        extractor.raw_triples.extend(triples)
        seen_entities.extend([value for triple in triples for value in (triple.subject, triple.object)])
        print(f"Chunk {chunk['id']}: {len(triples)} triples")

    extractor.processing_state["chunks_processed"] = len(chunks)
    extractor.processing_state["total_raw_triples"] = len(extractor.raw_triples)
    extractor.cluster_entities(similarity_threshold=0.88)
    consolidated = extractor.consolidate_triples()
    graph = build_graph(consolidated)

    (output_dir / "ontology.json").write_text(json.dumps(ontology, indent=2, ensure_ascii=False), encoding="utf-8")
    extractor.export_to_json(str(output_dir / "triples.json"))
    (output_dir / "graph.json").write_text(json.dumps(graph, indent=2, ensure_ascii=False), encoding="utf-8")
    write_graphml(graph, output_dir / "graph.graphml")
    write_html(graph, output_dir / "graph_visualization.html")
    if args.neo4j:
        load_into_neo4j(graph)

    state = extractor.get_state_snapshot()
    print(json.dumps({"provider": client.provider, "model": client.model, **state, "nodes": len(graph["nodes"]), "edges": len(graph["edges"])}, indent=2))
    print(f"Outputs written to {output_dir.resolve()}")


if __name__ == "__main__":
    main()


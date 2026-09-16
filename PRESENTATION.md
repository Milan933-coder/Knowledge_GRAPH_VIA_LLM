# LLM-Driven Knowledge Graph — presentation notes

## One-sentence explanation

This project converts unstructured text into a structured knowledge graph by
using an LLM for ontology discovery and fact extraction, embeddings for entity
normalisation, and Neo4j for persistent graph storage and traversal.

## Five-minute demo

```bash
python -m pip install -r requirements.txt
python run_pipeline.py --input raw_data.txt --max-chunks 4
```

Then open `outputs/graph_visualization.html` in a browser.

To use the local model after Ollama is installed:

```bash
ollama pull qwen2.5:1.5b
```

Change `LLM_PROVIDER=ollama` in `.env` and run the same command again.

## Explain the flow

1. The document is split into overlapping chunks so the model fits within its context window.
2. The model proposes classes, properties, and relationship types.
3. Each chunk is converted into subject–predicate–object triples with confidence scores.
4. Similar entity names are grouped and duplicate triples are removed.
5. The result is exported as JSON, GraphML, and an interactive HTML network.
6. With `--neo4j`, nodes and relationships are also loaded into Neo4j.

## Honest limitations

The current demo processes a configurable number of chunks and is designed for
clarity and presentation speed. Production use should add human review for
entity merges, source-span provenance, evaluation metrics, and incremental
processing for large document collections.


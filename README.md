🚀 LLM-Driven Knowledge Graph Pipeline
Ontology Engineering • Knowledge Extraction • Entity Resolution

An end-to-end experimental pipeline for building high-quality Knowledge Graphs (KGs) using small/efficient LLMs with dynamic memory handling, ontology discovery, and Neo4j-based entity resolution.

📌 Motivation

Building Knowledge Graphs from large corpora using LLMs faces three major challenges:

❗ Context window limitations

❗ Duplicate entity creation

❗ Weak ontology consistency

❗ Lack of dynamic memory across chunks

This project explores a modular, research-oriented pipeline that addresses these issues through:

✅ Ontology Engineering with LLMs
✅ Structured Knowledge Extraction
✅ Vector-based Entity Resolution
✅ Neo4j Graph Storage
✅ Dynamic memory strategies

🧠 High-Level Pipeline
flowchart LR
    A[Raw Corpus / PDF] --> B[Chunking]
    B --> C[Ontology Discovery]
    C --> D[Ontology Refinement]
    D --> E[Knowledge Extraction]
    E --> F[Entity Resolution]
    F --> G[Neo4j Graph]
    G --> H[Post-processing & Analysis]

🏗️ Project Architecture
.
├── ontology_extractor.py        # Ontology discovery & refinement
├── kg_llm_extractor.py          # LLM-based knowledge extraction
├── neo4j_entity_resolution.py   # Vector-based entity deduplication
├── .env                         # Neo4j credentials
└── README.md

🔬 Core Components
1️⃣ Ontology Engineering

File: ontology_extractor.py

🎯 Goal

Automatically discover and refine the schema before KG construction.

🧩 Approach

Phase 1 — Ontology Discovery

The LLM analyzes corpus chunks to identify:

Candidate entity types

Relationship types

Attribute patterns

Domain concepts

Phase 2 — Ontology Refinement

The discovered ontology is:

Deduplicated

Normalized

Structured into a consistent schema

Validated for conflicts

✅ Why This Matters

Without ontology grounding:

KG becomes noisy

Relations become inconsistent

Downstream reasoning fails

This step provides schema stability before extraction.

2️⃣ Knowledge Extraction

File: kg_llm_extractor.py

🎯 Goal

Convert unstructured text into structured triples.

⚙️ Extraction Strategy

For each chunk:

Pass chunk to small LLM

Extract structured triples

Attach metadata (chunk_id, confidence, etc.)

Store intermediate results

🧱 Output Format
{
  "entities": [...],
  "relationships": [...],
  "source_chunk": "...",
  "confidence": 0.xx
}

🔑 Key Design Decisions
✅ Chunk-aware extraction

Each triple retains provenance.

✅ Schema-guided prompting

Extraction is constrained by discovered ontology.

✅ Small-model friendly

Designed to run with lightweight models (e.g., Mistral-7B).

3️⃣ Entity Resolution (Deduplication)

File: neo4j_entity_resolution.py

This is one of the most critical innovations in the pipeline.

🚨 Problem

When processing chunks independently:

Same entity appears multiple times

Graph becomes fragmented

Query quality degrades

💡 Solution

We implement vector-based entity resolution using:

Sentence embeddings

Similarity search

Neo4j vector index

Threshold-based merging

🔄 Resolution Flow
flowchart TD
    A[New Entity] --> B[Generate Embedding]
    B --> C[Search Similar Entities]
    C --> D{Similarity > Threshold?}
    D -->|Yes| E[Merge Entities]
    D -->|No| F[Create New Node]

🧠 Matching Strategy

We compare using:

Name similarity

Semantic embedding similarity

Optional attribute matching

This significantly reduces duplicate nodes.

🧩 Dynamic Memory Strategy

One major discussion during development was:

❓ How do we prevent the LLM from "forgetting" previous chunks?

❌ Naive Approach

Process chunks independently → leads to:

duplicate entities

inconsistent relations

ontology drift

✅ Our Hybrid Solution

We combine:

🔹 Pre-Extraction Memory

Ontology grounding

Schema constraints

🔹 Post-Extraction Memory

Entity resolution

Graph merging

Community clustering

🚀 Why This Works Better

Instead of forcing the LLM to remember everything (which is expensive and unreliable), we:

✔ Let the model work locally
✔ Fix globally via graph algorithms
✔ Maintain scalability

🗄️ Neo4j Integration

The graph layer provides:

Persistent memory

Efficient traversal

Vector similarity search

Graph analytics

🔧 Required Environment Variables

Create a .env file:

NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password

▶️ How to Run
1️⃣ Install Dependencies
pip install -r requirements.txt

2️⃣ Start Neo4j

Make sure Neo4j is running locally.

3️⃣ Run Ontology Discovery
python ontology_extractor.py

4️⃣ Run Knowledge Extraction
python kg_llm_extractor.py

5️⃣ Run Entity Resolution
python neo4j_entity_resolution.py

📊 Design Philosophy

This project follows several important principles:

🧠 LLMs are not memory systems

We avoid overloading context windows and instead use:

Graph memory

Vector search

Post-processing

⚡ Small models > giant models (for pipelines)

The system is optimized for:

Mistral-7B class models

Kaggle/consumer GPUs

Efficient inference

🔄 Graph post-processing is essential

High-quality KGs require:

deduplication

clustering

schema enforcement

—not just extraction.

🧪 Future Work

 Online ontology adaptation

 Streaming KG construction

 Community detection integration

 Temporal knowledge graphs

 Multi-agent extraction

 GraphRAG integration

🤝 Contributions

Contributions, ideas, and research discussions are welcome!

If you are working on:

KG construction

GraphRAG

Ontology learning

LLM pipelines

feel free to open an issue or PR.

⭐ Acknowledgment

This project is an experimental research effort exploring the intersection of:

Knowledge Graphs

Small Language Models

Graph Databases

Representation Learning

💬 Author Note

Building robust Knowledge Graphs with LLMs is not just an extraction problem —
it is a systems engineering problem involving memory, ontology, and graph intelligence.

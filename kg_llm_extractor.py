#!/usr/bin/env python3
"""
LLM-Integrated Stateful Knowledge Graph Extractor
Uses local Qwen models with 8-bit quantization for intelligent triple extraction
"""

import json
from typing import List, Dict, Optional
import os
from kg_extractor import StatefulKGExtractor, Triple

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
except ImportError:
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
    BitsAndBytesConfig = None


class LLMStatefulKGExtractor(StatefulKGExtractor):
    """
    Extended KG extractor with LLM integration
    Maintains conversation state across chunks for context
    """
    
    def __init__(self, model_id: Optional[str] = None):
        super().__init__()
        self.model_id = model_id or os.getenv("QWEN_MODEL_ID", "Qwen/Qwen2.5-3B-Instruct")
        self.default_max_tokens = int(os.getenv("LLM_MAX_NEW_TOKENS", "768"))
        self.temperature = float(os.getenv("LLM_TEMPERATURE", "0.1"))
        self.max_context_tokens = int(os.getenv("LLM_MAX_CONTEXT_TOKENS", "4096"))
        self.model = None
        self.tokenizer = None
        self.client = self._load_local_model_8bit()
        
        # Stateful conversation memory

        self.conversation_history = []
        self.entity_context = {}  # Track entities seen so far for context
        self.domain_context = ""  # Overall document domain/topic

    def _load_local_model_8bit(self) -> bool:
        """Load a small Qwen model locally using 8-bit quantization."""
        if not all([torch, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig]):
            print(
                "Missing local LLM dependencies. Install with: "
                "pip install torch transformers bitsandbytes accelerate"
            )
            return False

        try:
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                quantization_config=quant_config,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
            print(f"Loaded local model: {self.model_id} (8-bit)")
            return True
        except Exception as exc:
            print(f"Failed to load local model '{self.model_id}': {exc}")
            return False

    def _chat_completion(self, prompt: str, max_tokens: int) -> str:
        """
        Run local Qwen inference using 8-bit model.
        """
        if not self.client:
            raise RuntimeError("No local Qwen model configured")

        tokens_to_generate = max(1, min(max_tokens, self.default_max_tokens))
        system_msg = (
            "You are a careful information extraction assistant. "
            "Follow JSON output instructions exactly."
        )
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ]

        if hasattr(self.tokenizer, "apply_chat_template"):
            generation_inputs = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(self.model.device)
        else:
            fallback_prompt = f"System: {system_msg}\nUser: {prompt}\nAssistant:"
            encoded = self.tokenizer(
                fallback_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_context_tokens,
            )
            generation_inputs = encoded["input_ids"].to(self.model.device)

        generate_kwargs = {
            "input_ids": generation_inputs,
            "max_new_tokens": tokens_to_generate,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "do_sample": self.temperature > 0,
        }
        if self.temperature > 0:
            generate_kwargs["temperature"] = self.temperature
            generate_kwargs["top_p"] = 0.9

        with torch.inference_mode():
            outputs = self.model.generate(**generate_kwargs)

        output_tokens = outputs[0][generation_inputs.shape[-1]:]
        return self.tokenizer.decode(output_tokens, skip_special_tokens=True).strip()
        
    def set_domain_context(self, text_sample: str):
        """
        Analyze document to understand domain/topic
        This helps with better entity recognition across chunks
        """
        if not self.client:
            return
        
        prompt = f"""Analyze this research paper excerpt and identify:
1. The main research domain (e.g., "Machine Learning", "Quantum Computing", "Bioinformatics")
2. Key terminology and concepts that might appear throughout
3. Common entity patterns (e.g., algorithm names, author citations, techniques)

Excerpt:
{text_sample[:1000]}

Respond in JSON format:
{{
    "domain": "...",
    "key_concepts": ["...", "..."],
    "entity_patterns": ["...", "..."]
}}
"""
        
        try:
            response_text = self._chat_completion(prompt=prompt, max_tokens=1000)
            
            result = json.loads(response_text)
            self.domain_context = result.get("domain", "")
            self.entity_context['key_concepts'] = set(result.get("key_concepts", []))
            
            print(f"Domain identified: {self.domain_context}")
            
        except Exception as e:
            print(f"Error in domain analysis: {e}")
    
    def extract_from_chunk_with_llm(self, chunk: Dict) -> List[Triple]:
        """
        Extract triples using LLM with stateful context
        
        Key features:
        - Passes previously seen entities for consistency
        - Uses domain context for better recognition
        - Maintains conversation history for coherence
        """
        if not self.client:
            print("No API client configured, falling back to rule-based extraction")
            return self.extract_from_chunk(chunk, use_llm=False)
        
        chunk_id = chunk['id']
        text = chunk['text']
        
        # Build context-aware prompt
        prompt = self._build_extraction_prompt(text, chunk_id)
        
        try:
            response_text = self._chat_completion(prompt=prompt, max_tokens=2000)
            
            # Parse response
            triples = self._parse_llm_response(response_text, chunk_id)
            
            # Update stateful memory
            self._update_entity_context(triples)
            self.conversation_history.append({
                'chunk_id': chunk_id,
                'triples_extracted': len(triples)
            })
            
            # Update state
            self.raw_triples.extend(triples)
            self.processing_state['chunks_processed'] += 1
            self.processing_state['total_raw_triples'] = len(self.raw_triples)
            
            return triples
            
        except Exception as e:
            print(f"Error in LLM extraction for chunk {chunk_id}: {e}")
            return []
    
    def _build_extraction_prompt(self, text: str, chunk_id: int) -> str:
        """Build context-aware extraction prompt"""
        
        # Get previously seen entities for consistency
        seen_entities = list(self.entity_context.get('key_concepts', []))[:20]
        
        prompt = f"""You are extracting knowledge graph triples from a research paper in the domain of: {self.domain_context or 'General Science'}

CHUNK {chunk_id} TEXT:
{text}

CONTEXT FROM PREVIOUS CHUNKS:
- Previously identified entities: {', '.join(seen_entities) if seen_entities else 'None yet'}
- Total chunks processed: {self.processing_state['chunks_processed']}

TASK:
Extract knowledge graph triples (subject, predicate, object) that capture:
1. Core concepts and their definitions
2. Relationships between concepts/methods/techniques
3. Authorship and citations
4. Experimental results and findings
5. Methodological relationships

GUIDELINES:
- Use consistent entity names (check previously identified entities)
- Extract both explicit and implicit relationships
- Use formal, standardized entity names
- Focus on research-relevant information
- Assign confidence (0.0-1.0) based on clarity

IMPORTANT:
- For entities you've seen before, use the EXACT same name
- If you see a variant (e.g., "ML" vs "Machine Learning"), note it but use the formal form

Respond ONLY with a JSON array:
[
  {{
    "subject": "entity or concept",
    "predicate": "relationship_type",
    "object": "related entity or value",
    "confidence": 0.0-1.0
  }}
]

Extract 5-15 triples. Focus on quality over quantity."""
        
        return prompt
    
    def _parse_llm_response(self, response_text: str, chunk_id: int) -> List[Triple]:
        """Parse LLM response into Triple objects"""
        try:
            # Clean up response (remove markdown if present)
            response_text = response_text.strip()
            if response_text.startswith('```'):
                response_text = response_text.split('```')[1]
                if response_text.startswith('json'):
                    response_text = response_text[4:]
            
            triples_data = json.loads(response_text)
            
            triples = []
            for item in triples_data:
                triple = Triple(
                    subject=item['subject'].strip(),
                    predicate=item['predicate'].strip(),
                    object=item['object'].strip(),
                    source_chunk=chunk_id,
                    confidence=float(item.get('confidence', 0.8))
                )
                triples.append(triple)
            
            return triples
            
        except json.JSONDecodeError as e:
            print(f"Error parsing LLM response: {e}")
            print(f"Response: {response_text[:200]}...")
            return []
    
    def _update_entity_context(self, triples: List[Triple]):
        """Update stateful entity context from newly extracted triples"""
        if 'key_concepts' not in self.entity_context:
            self.entity_context['key_concepts'] = set()
        
        for triple in triples:
            self.entity_context['key_concepts'].add(triple.subject)
            self.entity_context['key_concepts'].add(triple.object)
        
        # Keep only top N most frequent
        if len(self.entity_context['key_concepts']) > 100:
            self.entity_context['key_concepts'] = set(
                list(self.entity_context['key_concepts'])[:100]
            )
    
    def consolidate_with_llm_verification(self, sample_size: int = 5) -> List[Triple]:
        """
        Enhanced REDUCE phase: Use LLM to verify entity clusters
        
        For ambiguous cases, ask the local model to decide if entities should merge
        """
        if not self.client:
            return self.consolidate_triples()
        
        # First do standard consolidation
        self.consolidate_triples()
        
        # Then verify ambiguous clusters
        ambiguous_clusters = self._find_ambiguous_clusters()
        
        if ambiguous_clusters and len(ambiguous_clusters) <= sample_size:
            print(f"\nVerifying {len(ambiguous_clusters)} ambiguous entity clusters...")
            
            for canonical, variants in ambiguous_clusters:
                if self._should_merge_entities_llm(canonical, variants):
                    print(f"  ✓ Confirmed: {variants} -> '{canonical}'")
                else:
                    print(f"  ✗ Split cluster: {variants}")
                    # Don't merge these
                    self._unmerge_cluster(canonical, variants)
        
        return self.consolidated_triples
    
    def _find_ambiguous_clusters(self) -> List[tuple[str, set]]:
        """Find entity clusters that might be false positives"""
        ambiguous = []
        
        for canonical, variants in self.entity_clusters.items():
            if len(variants) <= 1:
                continue
            
            # Check if similarity is borderline (0.80-0.90)
            max_sim = max(
                self._entity_similarity(canonical, v) 
                for v in variants if v != canonical
            )
            
            if 0.80 <= max_sim <= 0.90:
                ambiguous.append((canonical, variants))
        
        return ambiguous
    
    def _should_merge_entities_llm(self, canonical: str, variants: set) -> bool:
        """Ask LLM if entities should be merged"""
        variants_list = list(variants)
        
        prompt = f"""In the context of a research paper about {self.domain_context}, 
should these entity names be merged into one?

Variants: {variants_list}
Proposed canonical name: "{canonical}"

Consider:
- Do they refer to the same concept/entity?
- Are they just different phrasings?
- Or are they genuinely different things?

Respond with ONLY: "MERGE" or "SPLIT"
"""
        
        try:
            response_text = self._chat_completion(prompt=prompt, max_tokens=50)
            
            decision = response_text.strip().upper()
            return "MERGE" in decision
            
        except Exception as e:
            print(f"Error in LLM verification: {e}")
            return True  # Default to merge
    
    def _unmerge_cluster(self, canonical: str, variants: set):
        """Separate a merged cluster back into individual entities"""
        # This would require re-processing triples
        # For now, just log
        pass


# ============= EXAMPLE USAGE =============

def example_with_llm():
    """Example using actual LLM integration"""
    
    # Sample research paper
    paper_text = """
    Knowledge Graphs (KGs) have long served as a fundamental infrastructure
for structured knowledge representation and reasoning. With the advent of
Large Language Models (LLMs), the construction of KGs has entered a new
paradigm—shifting from rule-based and statistical pipelines to language-driven
and generative frameworks. This survey provides a comprehensive overview of
recent progress in LLM-empowered knowledge graph construction, systemati
cally analyzing how LLMsreshapetheclassical three-layered pipeline of ontology
engineering, knowledge extraction, and knowledge fusion.
Wefirst revisit traditional KG methodologies to establish conceptual foundations,
and then review emerging LLM-driven approaches from two complementary per
spectives: schema-based paradigms, which emphasize structure, normalization,
and consistency; and schema-free paradigms, which highlight flexibility, adapt
ability, and open discovery. Across each stage, we synthesize representative
frameworks, analyze their technical mechanisms, and identify their limitations.
Finally, the survey outlines key trends and future research directions, including
KG-based reasoning for LLMs, dynamic knowledge memory for agentic systems,
and multimodal KG construction. Through this systematic review, we aim to clar
ify the evolving interplay between LLMs and knowledge graphs, bridging sym
bolic knowledge engineering and neural semantic understanding toward the de
velopment of adaptive, explainable, and intelligent knowledge systems.
1 INTRODUCTION
Knowledge Graphs (KGs) have long served as a cornerstone for representing, integrating, and rea
soning over structured knowledge. They provide a unified semantic foundation that underpins a wide
range of intelligent applications, such as semantic search, question answering, and scientific discov
ery. Conventional KG construction pipelines are typically composed of three major components:
ontology engineering, knowledge extraction, and knowledge fusion. Despite their success in en
abling large-scale knowledge organization, traditional paradigms (e.g., Zhong et al. (2023); Zhao
et al. (2024)) continue to face three enduring challenges: (1) Scalability and data sparsity, as rule
based and supervised systems often fail to generalize across domains; (2) Expert dependency and
rigidity, since schema and ontology design require substantial human intervention and lack adapt
ability; and (3) Pipeline fragmentation, where the disjoint handling of construction stages causes cu
mulative error propagation. These limitations hinder the development of self-evolving, large-scale,
and dynamic KGs.
The advent of Large Language Models (LLMs) introduces a transformative paradigm for over
coming these bottlenecks. Through large-scale pretraining and emergent generalization capabilities,
LLMs enable three key mechanisms: (1) Generative knowledge modeling, synthesizing structured
representations directly from unstructured text; (2) Semantic unification, integrating heterogeneous
knowledge sources through natural language grounding; and (3) Instruction-driven orchestration,
coordinating complex KG construction workflows via prompt-based interaction. Consequently,
LLMs are evolving beyond traditional text-processing tools into cognitive engines that seamlessly
bridge natural language and structured knowledge (e.g., Zhu et al. (2024b); Zhang & Soh (2024)).
1
Published as a conference paper at ICAIS 2025
This evolution marks a paradigm shift from rule-driven, pipeline-based systems toward LLM-driven,
unified, and adaptive frameworks, where knowledgeacquisition, organization, and reasoning emerge
as interdependent processes within a generative and self-refining ecosystem (Pan et al., 2024).
In light of these rapid advances, this paper presents a comprehensive survey of LLM-driven knowl
edge graph construction. We systematically review recent research spanning ontology engineering,
knowledge extraction, and fusion, analyze emerging methodological paradigms, and highlight open
challenges and future directions at the intersection of LLMs and knowledge representation.
The remainder of this paper is organized as follows:
• Section 2 introduces the foundations of traditional knowledge graph construction, covering
ontology engineering, knowledge extraction, and fusion techniques prior to the LLM era.
• Section 3 reviews LLM-enhanced ontology construction, encompassing both top-down
paradigms (LLMs as ontology assistants) and bottom-up paradigms (KGs for LLMs).
• Section 4 presents LLM-driven knowledge extraction, comparing schema-based and
schema-free methodologies.
• Section 5 discusses LLM-powered knowledge fusion, focusing on schema-level, instance
level, and hybrid frameworks.
• Section 6 explores future research directions, including KG-based reasoning, dynamic
knowledge memory, and multimodal KG construction.
2 PRELIMINARIES
The construction of Knowledge Graphs (KGs) traditionally follows a three-layered pipeline com
prising ontology engineering, knowledge extraction, and knowledge fusion. Prior to the advent of
Large Language Models (LLMs), these stages were implemented through rule-based, statistical, and
symbolic approaches. This section briefly reviews these conventional methodologies to establish
context for the subsequent discussion on LLM-empowered KG construction.
2.1 ONTOLOGY ENGINEERING
Ontology Engineering (OE) involves the formal specification of domain concepts, relationships, and
constraints. In the pre-LLM era, ontologies were primarily manually constructed by domain ex
perts, often supported by semantic web tools such as Prot´eg´e and guided by established methodolo
gies including METHONTOLOGY and On-To-Knowledge. These systematic processes emphasized
conceptual rigor and logical consistency but required extensive expert intervention.
As summarized by Zouaq & Nkambou (2010), ontology design during this period was char
acterized by strong human supervision and limited scalability. Subsequent semi-automatic ap
proaches—collectively known as ontology learning—sought to derive ontological structures from
textual corpora, as reviewed in Asim et al. (2018). However, even advanced frameworks such as
NeOn struggled with ontology evolution, modular reuse, and dynamic adaptation. As highlighted
by Kotis et al. (2020), traditional OE frameworks offered precision and formal soundness but lacked
f
lexibility and efficiency for large-scale or continuously evolving knowledge domains.
2.2 KNOWLEDGE EXTRACTION
KnowledgeExtraction (KE) aims to identify entities, relations, and attributes from unstructured or
semi-structured data. Early approaches relied on handcrafted linguistic rules and pattern matching,
which provided interpretability but were brittle and domain-specific. The evolution from symbolic
and rule-based systems to statistical and neural methods has been systematically summarized in Pai
et al. (2024).
The advent of deep learning architectures, such as BiLSTM-CRF and Transformer-based models,
marked a paradigm shift toward data-driven feature learning, as discussed by Yang et al. (2022b).
Comprehensive analyses such as Detroja et al. (2023) further categorize supervised, weakly super
vised, and unsupervised relation extraction paradigms, emphasizing their dependence on annotated
data and limited cross-domain generalization.
2
Published as a conference paper at ICAIS 2025
In summary, traditional KE methods established the technical foundation for modern extraction
pipelines but remained constrained by data scarcity, weak generalization, and cumulative error
propagation—limitations that motivate the LLM-driven paradigms discussed in later sections.
2.3 KNOWLEDGE FUSION
Knowledge Fusion (KF) focuses on integrating heterogeneous knowledge sources into a coherent
and consistent graph by resolving issues of duplication, conflict, and heterogeneity. A central sub
task is entity alignment, which determines whether entities from different datasets refer to the same
real-world object.
Classical approaches relied on lexical and structural similarity measures, as reviewed in Zeng et al.
(2021). The introduction of representation learning enabled embedding-based techniques that align
entities within shared vector spaces, improving scalability and automation, as surveyed by Zhu et al.
(2024a). Domain-specific applications, such as Yang et al. (2022a), demonstrate multi-feature fusion
strategies combining structural, attribute, and relational similarities. Other graph-level models, such
as Liu et al. (2022), further integrate semantic cues to enhance alignment robustness.
Despite these advancements, traditional fusion pipelines continue to struggle with semantic hetero
geneity, large-scale integration, and dynamic knowledge updating—challenges that contempo
rary LLM-based fusion frameworks are increasingly designed to address.
Figure 1: Taxonomy of LLM for KGC
3 LLM-ENHANCED ONTOLOGY CONSTRUCTION
The integration of Large Language Models (LLMs) has introduced a fundamental paradigm shift
in Ontology Engineering (OE) and, by extension, Knowledge Graph (KG) construction. Current
research generally follows two complementary directions: a top-down approach, which leverages
LLMs as intelligent assistants for formal ontology modeling, and a bottom-up approach, which
employs ontology construction to enhance the reasoning and representation capabilities of LLMs
themselves.
3.1 TOP-DOWN ONTOLOGY CONSTRUCTION: LLMS AS ONTOLOGY ASSISTANTS
The top-down paradigm extends the traditions of the Semantic Web and Knowledge Engineering,
emphasizing ontology development guided by predefined semantic requirements. Within this frame
work, LLMsserveasadvancedco-modelers that assist human experts in translating natural language
specifications—such as competency questions (CQs), user stories, or domain descriptions—into
formal ontologies, typically represented in OWL or related standards. This paradigm prioritizes
conceptual abstraction, the precise definition of relations, and structured semantic representation to
3
Published as a conference paper at ICAIS 2025
ensure that subsequent knowledge extraction and instance population adhere to well-defined logical
constraints.
3.1.1 COMPETENCY QUESTION (CQ)-BASED ONTOLOGY CONSTRUCTION
CQ-based methods represent a requirements-driven pathway toward automated ontology modeling.
In this setting, LLMs parse CQs or user stories to identify, categorize, and formalize domain-specific
concepts, attributes, and relationships.
A pioneering framework, Ontogenia (Lippolis et al., 2025a), introduced the use of Metacogni
tive Prompting for ontology generation, enabling the model to perform self-reflection and struc
tural correction during synthesis. By incorporating Ontology Design Patterns (ODPs), Ontogenia
improves both the consistency and complexity of generated ontologies. Similarly, the CQbyCQ
framework (Saeedizade & Blomqvist, 2024) demonstrated that LLMs can directly translate CQs and
user stories into OWL-compliant schemas, effectively automating the transition from requirements
to structured ontological models.
Building on these advances, Lippolis et al. (2025b) proposed two complementary prompting strate
gies: a “memoryless” approach for modular construction and a reflective iterative method inspired
by Ontogenia. Empirical evaluations revealed that LLMs can autonomously identify classes, ob
ject properties, and data properties, while generating corresponding logical axioms with consistency
comparable to that of junior human modelers. Collectively, these studies have led to semi-automated
ontology construction pipelines encompassing the entire lifecycle—from CQ formulation and val
idation to ontology instantiation—with human experts intervening only at critical checkpoints.
Through this evolution, LLMs have transitioned from passive analytical tools to active modeling
collaborators in ontology design.
3.1.2 NATURAL LANGUAGE-BASED ONTOLOGY CONSTRUCTION
BeyondCQ-drivenparadigms, natural language-based ontology construction seeks to induce seman
tic schemas directly from unstructured or semi-structured text corpora, eliminating the dependency
on explicitly formulated questions. The goal is to enable LLMs to autonomously uncover conceptual
hierarchies and relational patterns from natural language, achieving a direct mapping from textual
descriptions to formal logical representations.
Foundational work in this domain—including Saeedizade & Blomqvist (2024) and Lippolis et al.
(2025b)—systematically evaluated GPT-4’s performance and confirmed that its outputs approach the
quality of novice human modelers, thereby validating the feasibility of “intelligent ontology assis
tants.” The LLMs4OL framework (Giglou et al., 2023) further verified LLMs’ capacity for concept
identification, relation extraction, and semantic pattern induction in general-purpose domains. Like
wise, Mateiu & Groza (2023) demonstrated the use of fine-tuned models to directly translate natural
language into OWL axioms within established ontology editors such as Prot´ eg´ e.
Recent systems such as NeOn-GPT (Fathallah et al., 2025) and LLMs4Life (Fathallah et al.,
2024) have advanced this direction by introducing end-to-end, prompt-driven workflows that in
tegrate ontology reuse and adaptive refinement to construct deep, coherent ontological structures
in complex scientific domains (e.g., life sciences). Meanwhile, lightweight frameworks such as
LKD-KGC(Sunetal., 2025) enable rapid schema induction for open-domain knowledge graphs by
clustering entity types extracted from document summaries.
In summary, top-down research on LLM-assisted ontology construction emphasizes semantic con
sistency, structural completeness, and human–AI collaboration, marking a significant evolution of
traditional knowledge engineering toward more intelligent, language-driven paradigms.
3.2 BOTTOM-UP ONTOLOGY SCHEMA CONSTRUCTION: KGS FOR LLMS
The bottom-up methodology has gained increasing attention as a response to paradigm shifts intro
duced by the era of Large Language Models (LLMs), particularly within Retrieval-Augmented Gen
eration (RAG) frameworks. In this paradigm, the knowledge graph is no longer viewed merely as a
static repository of structured knowledge for human interpretation. Instead, it serves as a dynamic
infrastructure that provides factual grounding and structured memory for LLMs. Consequently, re
4
Published as a conference paper at ICAIS 2025
search focus has shifted from manually designing ontological hierarchies to automatically inducing
schemas from unstructured or semi-structured data. This evolution can be delineated through three
interrelated stages of progress.
Early studies such as GraphRAG(Edgeetal., 2024) and OntoRAG(Tiwarietal., 2025) established
the foundation for data-driven ontology construction. These approaches first generate instance-level
graphs from raw text via open information extraction, and then abstract ontological concepts and
relations through clustering and generalization. This “data-to-schema” process transforms empirical
knowledge into reusable conceptual structures, illustrating how instance-rich corpora can give rise
to ontological blueprints.
Building upon this foundation, the EDC (Extract–Define–Canonicalize) framework (Zhang &Soh,
2024) advanced the pipeline into a three-stage process consisting of open extraction, semantic defi
nition, and schema normalization. It enables the alignment of automatically induced schemas with
existing ontologies, or the creation of new ones when predefined structures are absent. Extending
this adaptability, AdaKGC (Ye et al., 2023) addressed the challenge of dynamic schema evolution,
allowing models to incorporate novel relations and entity types without retraining. Collectively,
these advances shift the focus from static schema construction toward continuous schema adapta
tion within evolving knowledge environments.
More recent efforts have transitioned beyond algorithmic prototypes toward deployable knowledge
systems. For example, AutoSchemaKG (Bai et al., 2025) integrates schema-based and schema
free paradigms within a unified architecture, supporting the real-time generation and evolution of
enterprise-scale knowledge graphs. In this stage, KGs operate as a form of external knowledge
memory for LLMs—prioritizing factual coverage, scalability, and maintainability over purely se
mantic completeness. This transformation marks a pragmatic reorientation of ontology construction,
emphasizing its service to LLM reasoning and interpretability in knowledge-intensive applications.
In summary, bottom-up ontology schema construction redefines the interplay between LLMs and
knowledge engineering. The focus evolves from “LLMs for Ontology Engineering” to “Ontolo
gies and KGsforLLMs”. Whereasthetop-downtrajectory emphasizes semantic modeling, logical
consistency, and expert-guided alignment—positioning LLMs as intelligent assistants in ontology
design—the bottom-up trajectory prioritizes automatic extraction, schema induction, and dynamic
evolution. This progression advances toward self-updating, interpretable, and scalable knowledge
ecosystems that strengthen the grounding and reasoning capabilities of LLMs
    """
    
    # Initialize local small Qwen model (8-bit quantized by default)
    # Override with env var: QWEN_MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct
    extractor = LLMStatefulKGExtractor()
    
    print("=" * 70)
    print("LLM-INTEGRATED STATEFUL KG EXTRACTION")
    print("=" * 70)
    
    # Set domain context
    print("\n[0] Analyzing document domain...")
    extractor.set_domain_context(paper_text)
    
    # Chunk document
    print("\n[1] Chunking document...")
    chunks = extractor.chunk_document(paper_text, chunk_size=80, overlap=15)
    print(f"Created {len(chunks)} chunks")
    
    # Extract with stateful context
    print("\n[2] MAP PHASE: Extracting with LLM (stateful)...")
    for chunk in chunks:
        triples = extractor.extract_from_chunk_with_llm(chunk)
        print(f"  Chunk {chunk['id']}: {len(triples)} triples")
    
    # Cluster entities
    print("\n[3] SHUFFLE PHASE: Clustering entities...")
    clusters = extractor.cluster_entities(similarity_threshold=0.82)
    print(f"Created {len(clusters)} entity clusters")
    
    # Consolidate with LLM verification
    print("\n[4] REDUCE PHASE: Consolidating with LLM verification...")
    consolidated = extractor.consolidate_with_llm_verification()
    
    # Results
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    state = extractor.get_state_snapshot()
    print(f"Raw triples: {state['raw_triples']}")
    print(f"Consolidated triples: {state['consolidated_triples']}")
    print(f"Deduplication: {state['deduplication_ratio']:.1%}")
    
    # Export
    extractor.export_to_json("kg_llm_export.json")
    
    return extractor


if __name__ == "__main__":
    example_with_llm()

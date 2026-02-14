"""
Ontology Engineering System using Local LLM
Extracts ontology from text corpus and refines it by removing duplicates
"""

import json
import re
from typing import List, Dict, Set, Tuple
from collections import defaultdict
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import requests


class LocalLLMOntologyExtractor:
    """Extract ontology using local LLM (Ollama)"""
    
    def __init__(self, model_name: str = "llama2", ollama_url: str = "http://localhost:11434"):
        """
        Initialize the ontology extractor
        
        Args:
            model_name: Name of the Ollama model to use (llama2, mistral, etc.)
            ollama_url: URL where Ollama is running
        """
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.api_endpoint = f"{ollama_url}/api/generate"
        
    def call_llm(self, prompt: str, temperature: float = 0.3) -> str:
        """Call local LLM via Ollama API"""
        try:
            response = requests.post(
                self.api_endpoint,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "temperature": temperature,
                    "stream": False
                },
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                print(f"Error calling LLM: {response.status_code}")
                return ""
        except Exception as e:
            print(f"Error calling LLM: {str(e)}")
            return ""
    
    def extract_ontology_from_text(self, text: str) -> Dict:
        """Extract ontology elements from text using LLM"""
        
        prompt = f"""You are an ontology engineering expert. Analyze the following text and extract ontology elements in JSON format.

Extract:
1. Classes (main concepts/entities)
2. Properties (attributes of classes)
3. Relationships (connections between classes)

Text:
{text}

Provide your response ONLY as a valid JSON object with this structure:
{{
  "classes": ["Class1", "Class2", ...],
  "properties": {{
    "Class1": ["property1", "property2", ...],
    "Class2": ["property1", ...]
  }},
  "relationships": [
    {{"subject": "Class1", "predicate": "relationName", "object": "Class2"}},
    ...
  ]
}}

JSON Output:"""

        response = self.call_llm(prompt)
        
        # Parse JSON from response
        try:
            # Try to find JSON in the response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                ontology = json.loads(json_match.group())
                return ontology
            else:
                print("No JSON found in response")
                return self._create_empty_ontology()
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            print(f"Response: {response[:200]}")
            return self._create_empty_ontology()
    
    def extract_from_corpus(self, corpus: List[str], chunk_size: int = 1000) -> Dict:
        """
        Extract ontology from a corpus of documents
        
        Args:
            corpus: List of text documents
            chunk_size: Maximum characters per chunk
        
        Returns:
            Combined ontology dictionary
        """
        combined_ontology = {
            "classes": [],
            "properties": defaultdict(list),
            "relationships": []
        }
        
        print(f"Processing {len(corpus)} documents...")
        
        for idx, document in enumerate(corpus):
            print(f"Processing document {idx + 1}/{len(corpus)}...")
            
            # Split document into chunks if needed
            chunks = self._split_text(document, chunk_size)
            
            for chunk_idx, chunk in enumerate(chunks):
                print(f"  Chunk {chunk_idx + 1}/{len(chunks)}...")
                ontology = self.extract_ontology_from_text(chunk)
                
                # Merge with combined ontology
                combined_ontology["classes"].extend(ontology.get("classes", []))
                
                for cls, props in ontology.get("properties", {}).items():
                    combined_ontology["properties"][cls].extend(props)
                
                combined_ontology["relationships"].extend(
                    ontology.get("relationships", [])
                )
        
        # Convert defaultdict to regular dict
        combined_ontology["properties"] = dict(combined_ontology["properties"])
        
        return combined_ontology
    
    def _split_text(self, text: str, chunk_size: int) -> List[str]:
        """Split text into chunks"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 > chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0
            current_chunk.append(word)
            current_length += len(word) + 1
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def _create_empty_ontology(self) -> Dict:
        """Create empty ontology structure"""
        return {
            "classes": [],
            "properties": {},
            "relationships": []
        }


class OntologyRefiner:
    """Refine ontology by removing duplicates and merging similar concepts"""
    
    def __init__(self, similarity_threshold: float = 0.85):
        """
        Initialize the refiner
        
        Args:
            similarity_threshold: Threshold for considering items as duplicates (0-1)
        """
        self.similarity_threshold = similarity_threshold
        # Use a lightweight sentence transformer model
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Embedding model loaded.")
    
    def refine_ontology(self, ontology: Dict) -> Dict:
        """
        Refine ontology by removing duplicates and merging similar concepts
        
        Args:
            ontology: Raw ontology dictionary
            
        Returns:
            Refined ontology dictionary
        """
        print("\n=== Starting Ontology Refinement ===")
        
        refined = {
            "classes": [],
            "properties": {},
            "relationships": []
        }
        
        # 1. Refine classes
        print("\n1. Refining classes...")
        original_classes = list(set(ontology.get("classes", [])))
        print(f"   Original unique classes: {len(original_classes)}")
        
        refined["classes"], class_mapping = self._merge_similar_items(original_classes)
        print(f"   Refined classes: {len(refined['classes'])}")
        print(f"   Duplicates removed: {len(original_classes) - len(refined['classes'])}")
        
        # 2. Refine properties using class mapping
        print("\n2. Refining properties...")
        refined["properties"] = self._refine_properties(
            ontology.get("properties", {}),
            class_mapping
        )
        
        # 3. Refine relationships using class mapping
        print("\n3. Refining relationships...")
        refined["relationships"] = self._refine_relationships(
            ontology.get("relationships", []),
            class_mapping
        )
        
        print("\n=== Refinement Complete ===\n")
        
        return refined
    
    def _merge_similar_items(self, items: List[str]) -> Tuple[List[str], Dict[str, str]]:
        """
        Merge similar items using semantic similarity
        
        Returns:
            (merged_items, mapping_dict) where mapping maps old items to canonical forms
        """
        if not items:
            return [], {}
        
        # Clean and normalize items
        cleaned_items = [self._clean_text(item) for item in items]
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(cleaned_items)
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        
        # Find clusters of similar items
        merged = []
        mapping = {}
        used = set()
        
        for i, item in enumerate(cleaned_items):
            if i in used:
                continue
            
            # Find all similar items
            similar_indices = np.where(similarity_matrix[i] >= self.similarity_threshold)[0]
            cluster = [cleaned_items[idx] for idx in similar_indices if idx not in used]
            
            if cluster:
                # Choose canonical form (most common or shortest)
                canonical = self._choose_canonical(cluster)
                merged.append(canonical)
                
                # Map all items in cluster to canonical
                for idx in similar_indices:
                    if idx not in used:
                        mapping[cleaned_items[idx]] = canonical
                        used.add(idx)
        
        return merged, mapping
    
    def _refine_properties(self, properties: Dict, class_mapping: Dict) -> Dict:
        """Refine properties using class mapping"""
        refined_props = defaultdict(list)
        
        for cls, props in properties.items():
            # Map class to canonical form
            canonical_cls = class_mapping.get(self._clean_text(cls), cls)
            
            # Merge similar properties
            if props:
                merged_props, _ = self._merge_similar_items(props)
                refined_props[canonical_cls].extend(merged_props)
        
        # Remove duplicates within each class
        for cls in refined_props:
            refined_props[cls] = list(set(refined_props[cls]))
        
        return dict(refined_props)
    
    def _refine_relationships(self, relationships: List[Dict], class_mapping: Dict) -> List[Dict]:
        """Refine relationships using class mapping"""
        refined_rels = []
        seen = set()
        
        for rel in relationships:
            subject = rel.get("subject", "")
            predicate = rel.get("predicate", "")
            obj = rel.get("object", "")
            
            # Map to canonical forms
            canonical_subject = class_mapping.get(self._clean_text(subject), subject)
            canonical_object = class_mapping.get(self._clean_text(obj), obj)
            
            # Create unique key for deduplication
            rel_key = (canonical_subject, predicate.lower(), canonical_object)
            
            if rel_key not in seen:
                refined_rels.append({
                    "subject": canonical_subject,
                    "predicate": predicate,
                    "object": canonical_object
                })
                seen.add(rel_key)
        
        return refined_rels
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        # Remove extra whitespace, lowercase
        text = re.sub(r'\s+', ' ', text.strip())
        return text.lower()
    
    def _choose_canonical(self, items: List[str]) -> str:
        """Choose canonical form from similar items"""
        if not items:
            return ""
        
        # Prefer most common capitalization pattern
        # or shortest form
        return min(items, key=len)


class OntologyExporter:
    """Export ontology in various formats"""
    
    @staticmethod
    def to_json(ontology: Dict, filepath: str):
        """Export ontology as JSON"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(ontology, f, indent=2, ensure_ascii=False)
        print(f"Ontology exported to {filepath}")
    
    @staticmethod
    def to_rdf(ontology: Dict, filepath: str, namespace: str = "http://example.org/ontology#"):
        """Export ontology as RDF/Turtle format"""
        lines = [
            "@prefix : <{}> .".format(namespace),
            "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .",
            "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
            "@prefix owl: <http://www.w3.org/2002/07/owl#> .",
            "",
            "# Ontology definition",
            f"<{namespace}> rdf:type owl:Ontology .",
            ""
        ]
        
        # Export classes
        lines.append("# Classes")
        for cls in ontology.get("classes", []):
            safe_cls = cls.replace(" ", "_")
            lines.append(f":{safe_cls} rdf:type owl:Class .")
        lines.append("")
        
        # Export properties
        lines.append("# Properties")
        for cls, props in ontology.get("properties", {}).items():
            safe_cls = cls.replace(" ", "_")
            for prop in props:
                safe_prop = prop.replace(" ", "_")
                lines.append(f":{safe_prop} rdf:type owl:DatatypeProperty ;")
                lines.append(f"    rdfs:domain :{safe_cls} .")
        lines.append("")
        
        # Export relationships
        lines.append("# Relationships")
        for rel in ontology.get("relationships", []):
            subject = rel["subject"].replace(" ", "_")
            predicate = rel["predicate"].replace(" ", "_")
            obj = rel["object"].replace(" ", "_")
            lines.append(f":{predicate} rdf:type owl:ObjectProperty ;")
            lines.append(f"    rdfs:domain :{subject} ;")
            lines.append(f"    rdfs:range :{obj} .")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))
        print(f"Ontology exported to {filepath}")
    
    @staticmethod
    def print_summary(ontology: Dict):
        """Print ontology summary"""
        print("\n" + "="*60)
        print("ONTOLOGY SUMMARY")
        print("="*60)
        
        print(f"\nClasses ({len(ontology.get('classes', []))}):")
        for cls in sorted(ontology.get("classes", []))[:20]:
            print(f"  - {cls}")
        if len(ontology.get("classes", [])) > 20:
            print(f"  ... and {len(ontology.get('classes', [])) - 20} more")
        
        print(f"\nProperties ({sum(len(props) for props in ontology.get('properties', {}).values())}):")
        for cls, props in list(ontology.get("properties", {}).items())[:10]:
            print(f"  {cls}:")
            for prop in props[:5]:
                print(f"    - {prop}")
        
        print(f"\nRelationships ({len(ontology.get('relationships', []))}):")
        for rel in ontology.get("relationships", [])[:15]:
            print(f"  - {rel['subject']} --[{rel['predicate']}]--> {rel['object']}")
        if len(ontology.get("relationships", [])) > 15:
            print(f"  ... and {len(ontology.get('relationships', [])) - 15} more")
        
        print("\n" + "="*60 + "\n")


def main():
    """Main execution pipeline"""
    
    # Sample corpus for demonstration
    sample_corpus = [
        """
        A university is an educational institution that offers undergraduate and graduate programs.
        Students enroll in courses taught by professors. The university has multiple departments,
        each with its own faculty members. Students can major in various subjects like Computer Science,
        Biology, or Mathematics. Professors conduct research and publish papers in academic journals.
        """,
        """
        In a university setting, the library provides resources for students and faculty.
        Undergraduate students pursue bachelor degrees while graduate students work towards
        master's or doctoral degrees. Each course has a syllabus, assignments, and exams.
        Faculty members belong to academic departments and teach multiple courses per semester.
        """,
        """
        Research laboratories are part of universities where scientists and researchers
        conduct experiments. PhD students work as research assistants under the supervision
        of principal investigators. Universities also have administrative staff who manage
        operations. The registrar's office handles student enrollment and academic records.
        """
    ]
    
    print("="*60)
    print("ONTOLOGY ENGINEERING PIPELINE")
    print("="*60)
    
    # Step 1: Extract ontology using local LLM
    print("\n### STEP 1: ONTOLOGY EXTRACTION ###\n")
    extractor = LocalLLMOntologyExtractor(model_name="llama2")
    
    print("Extracting ontology from corpus...")
    raw_ontology = extractor.extract_from_corpus(sample_corpus)
    
    print("\n--- Raw Ontology Statistics ---")
    print(f"Total classes: {len(raw_ontology['classes'])}")
    print(f"Total properties: {sum(len(props) for props in raw_ontology.get('properties', {}).values())}")
    print(f"Total relationships: {len(raw_ontology['relationships'])}")
    
    # Step 2: Refine ontology
    print("\n### STEP 2: ONTOLOGY REFINEMENT ###\n")
    refiner = OntologyRefiner(similarity_threshold=0.85)
    refined_ontology = refiner.refine_ontology(raw_ontology)
    
    # Step 3: Export results
    print("\n### STEP 3: EXPORT ###\n")
    exporter = OntologyExporter()
    
    # Export raw ontology
    exporter.to_json(raw_ontology, "/home/claude/ontology_raw.json")
    
    # Export refined ontology
    exporter.to_json(refined_ontology, "/home/claude/ontology_refined.json")
    exporter.to_rdf(refined_ontology, "/home/claude/ontology_refined.ttl")
    
    # Print summaries
    print("\n" + "="*60)
    print("RAW ONTOLOGY")
    exporter.print_summary(raw_ontology)
    
    print("\n" + "="*60)
    print("REFINED ONTOLOGY")
    exporter.print_summary(refined_ontology)
    
    print("\n✓ Pipeline completed successfully!")
    print("\nGenerated files:")
    print("  - ontology_raw.json (raw extracted ontology)")
    print("  - ontology_refined.json (refined ontology)")
    print("  - ontology_refined.ttl (RDF/Turtle format)")


if __name__ == "__main__":
    main()

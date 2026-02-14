"""
Entity Resolution in Neo4j Vector Database
Uses local models and Neo4j's built-in features for deduplication
"""
import networkx as nx
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Tuple, Set
import json
from collections import defaultdict
from dotenv import load_dotenv
import os 
load_dotenv()

class Neo4jEntityResolver:
    """
    Entity Resolution in Neo4j using:
    - Vector embeddings from local model
    - Neo4j vector indexes
    - Neo4j's built-in similarity functions
    - Graph algorithms for deduplication
    """
    
    def __init__(self, uri: str, user: str, password: str, 
                 embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize Neo4j connection and embedding model
        
        Args:
            uri: Neo4j connection URI (e.g., "bolt://localhost:7687")
            user: Neo4j username
            password: Neo4j password
            embedding_model: Sentence transformer model name
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
        print(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        print(f"Model loaded. Embedding dimension: {self.embedding_dim}")
        
        # Initialize database
        self._setup_database()
    
    def close(self):
        """Close database connection"""
        self.driver.close()
    
    def _setup_database(self):
        """Setup constraints and indexes"""
        with self.driver.session() as session:
            # Create uniqueness constraint
            try:
                session.run("""
                    CREATE CONSTRAINT entity_id IF NOT EXISTS
                    FOR (e:Entity) REQUIRE e.id IS UNIQUE
                """)
                print("✓ Created uniqueness constraint")
            except Exception as e:
                print(f"Constraint already exists or error: {e}")
            
            # Create vector index for similarity search
            try:
                session.run(f"""
                    CREATE VECTOR INDEX entity_embeddings IF NOT EXISTS
                    FOR (e:Entity) ON (e.embedding)
                    OPTIONS {{
                        indexConfig: {{
                            `vector.dimensions`: {self.embedding_dim},
                            `vector.similarity_function`: 'cosine'
                        }}
                    }}
                """)
                print("✓ Created vector index")
            except Exception as e:
                print(f"Vector index setup: {e}")
            
            # Create text index for fuzzy matching
            try:
                session.run("""
                    CREATE FULLTEXT INDEX entity_names IF NOT EXISTS
                    FOR (e:Entity) ON EACH [e.name, e.normalized_name]
                """)
                print("✓ Created fulltext index")
            except Exception as e:
                print(f"Fulltext index: {e}")
    
    def add_entities(self, entities: List[Dict]):
        """
        Add entities to Neo4j with embeddings
        
        Args:
            entities: List of entity dicts with 'id', 'name', 'type', and optional 'properties'
        """
        print(f"\nAdding {len(entities)} entities to Neo4j...")
        
        with self.driver.session() as session:
            for entity in entities:
                entity_id = entity['id']
                name = entity['name']
                entity_type = entity.get('type', 'Entity')
                properties = entity.get('properties', {})
                
                # Generate embedding
                embedding = self.embedding_model.encode(name)
                
                # Normalize name for fuzzy matching
                normalized_name = self._normalize_text(name)
                
                # Create entity node
                session.run("""
                    MERGE (e:Entity {id: $id})
                    SET e.name = $name,
                        e.type = $type,
                        e.normalized_name = $normalized_name,
                        e.embedding = $embedding,
                        e.properties = $properties,
                        e.resolved = false
                """, {
                    'id': entity_id,
                    'name': name,
                    'type': entity_type,
                    'normalized_name': normalized_name,
                    'embedding': embedding.tolist(),
                    'properties': json.dumps(properties)
                })
        
        print(f"✓ Added {len(entities)} entities")
    


    def find_duplicate_clusters(self) -> List[Set[str]]:
        """
        Replaces GDS-based clustering with local NetworkX logic.
        Finds clusters of similar nodes using Weakly Connected Components.
        """
        clusters = []
        
        with self.driver.session() as session:
            # 1. Fetch all SIMILAR_TO relationships from the database
            # These were created in Step 2 of the pipeline
            query = """
            MATCH (n:Entity)-[:SIMILAR_TO]-(m:Entity)
            RETURN n.id AS source, m.id AS target
            """
            result = session.run(query)
            
            # 2. Build a local undirected graph
            G = nx.Graph()
            
            edge_count = 0
            for record in result:
                G.add_edge(record["source"], record["target"])
                edge_count += 1
                
            if edge_count == 0:
                print("No similarity edges found to cluster.")
                return []

            # 3. Run Weakly Connected Components (WCC) algorithm locally
            # This groups nodes where a path of SIMILAR_TO edges exists between them
            components = list(nx.connected_components(G))
            
            # 4. Convert to list of sets (matching the original script's expected format)
            clusters = [set(component) for component in components]
            
        print(f"Found {len(clusters)} duplicate clusters using local NetworkX WCC")
        return clusters    
    def find_duplicates_fuzzy(self) -> List[Dict]:
        """
        Find duplicates using Neo4j's fulltext search and fuzzy matching
        
        Returns:
            List of potential duplicates found via fuzzy matching
        """
        print("\nFinding duplicates using fuzzy matching...")
        
        candidates = []
        
        with self.driver.session() as session:
            # Use Levenshtein distance for fuzzy matching
            result = session.run("""
                MATCH (e1:Entity), (e2:Entity)
                WHERE e1.id < e2.id
                  AND (e1.resolved = false OR e1.resolved IS NULL)
                  AND (e2.resolved = false OR e2.resolved IS NULL)
                  AND apoc.text.levenshteinDistance(
                      toLower(e1.normalized_name), 
                      toLower(e2.normalized_name)
                  ) <= 3
                RETURN e1.id AS entity1_id,
                       e1.name AS entity1_name,
                       e2.id AS entity2_id,
                       e2.name AS entity2_name,
                       apoc.text.levenshteinDistance(
                           toLower(e1.normalized_name),
                           toLower(e2.normalized_name)
                       ) AS distance
                ORDER BY distance
            """)
            
            for record in result:
                candidates.append({
                    'entity1_id': record['entity1_id'],
                    'entity1_name': record['entity1_name'],
                    'entity2_id': record['entity2_id'],
                    'entity2_name': record['entity2_name'],
                    'levenshtein_distance': record['distance']
                })
        
        print(f"✓ Found {len(candidates)} fuzzy match candidates")
        return candidates
    
    def merge_duplicate_entities(self, entity1_id: str, entity2_id: str,
                                 keep_id: str = None):
        """
        Merge two duplicate entities using Neo4j's APOC merge
        
        Args:
            entity1_id: First entity ID
            entity2_id: Second entity ID
            keep_id: Which entity ID to keep (default: entity1_id)
        """
        keep_id = keep_id or entity1_id
        remove_id = entity2_id if keep_id == entity1_id else entity1_id
        
        with self.driver.session() as session:
            # Use APOC to merge nodes
            session.run("""
                MATCH (keep:Entity {id: $keep_id})
                MATCH (remove:Entity {id: $remove_id})
                
                // Merge relationships from remove to keep
                CALL apoc.refactor.mergeNodes([keep, remove], {
                    properties: 'combine',
                    mergeRels: true
                })
                YIELD node
                
                // Mark as resolved
                SET node.resolved = true,
                    node.merged_from = coalesce(node.merged_from, []) + [$remove_id]
                
                RETURN node.id AS merged_id
            """, {
                'keep_id': keep_id,
                'remove_id': remove_id
            })
        
        print(f"✓ Merged {remove_id} into {keep_id}")
    
    def auto_resolve_duplicates(self, similarity_threshold: float = 0.90):
        """
        Automatically resolve high-confidence duplicates
        
        Args:
            similarity_threshold: Only auto-merge above this threshold
        """
        print(f"\n=== Auto-resolving duplicates (threshold: {similarity_threshold}) ===")
        
        candidates = self.find_duplicate_candidates(similarity_threshold)
        
        merged_count = 0
        for candidate in candidates:
            if candidate['similarity'] >= similarity_threshold:
                try:
                    self.merge_duplicate_entities(
                        candidate['entity1_id'],
                        candidate['entity2_id']
                    )
                    merged_count += 1
                except Exception as e:
                    print(f"Error merging: {e}")
        
        print(f"\n✓ Auto-merged {merged_count} duplicate pairs")
    
    def create_similarity_edges(self, similarity_threshold: float = 0.80):
        """
        Create SIMILAR_TO relationships between similar entities
        Useful for manual review and clustering
        
        Args:
            similarity_threshold: Minimum similarity for creating edge
        """
        print(f"\nCreating similarity edges (threshold: {similarity_threshold})...")
        
        with self.driver.session() as session:
            result = session.run("""
                MATCH (e1:Entity), (e2:Entity)
                WHERE e1.id < e2.id
                  AND (e1.resolved = false OR e1.resolved IS NULL)
                  AND (e2.resolved = false OR e2.resolved IS NULL)
                WITH e1, e2,
                     gds.similarity.cosine(e1.embedding, e2.embedding) AS similarity
                WHERE similarity >= $threshold
                MERGE (e1)-[r:SIMILAR_TO]->(e2)
                SET r.similarity = similarity
                RETURN count(r) AS edges_created
            """, {'threshold': similarity_threshold})
            
            record = result.single()
            print(f"✓ Created {record['edges_created']} similarity edges")
    
    def find_duplicate_clusters(self) -> List[Set[str]]:
        """
        Find clusters of duplicate entities using graph algorithms
        Uses connected components on SIMILAR_TO relationships
        
        Returns:
            List of sets, each containing IDs of entities in a cluster
        """
        print("\nFinding duplicate clusters using graph algorithms...")
        
        with self.driver.session() as session:
            # Use Weakly Connected Components algorithm
            result = session.run("""
                CALL gds.wcc.stream({
                    nodeProjection: 'Entity',
                    relationshipProjection: {
                        SIMILAR_TO: {
                            type: 'SIMILAR_TO',
                            orientation: 'UNDIRECTED'
                        }
                    }
                })
                YIELD nodeId, componentId
                WITH gds.util.asNode(nodeId) AS entity, componentId
                WHERE entity.resolved = false OR entity.resolved IS NULL
                RETURN componentId, collect(entity.id) AS cluster_members
                ORDER BY size(cluster_members) DESC
            """)
            
            clusters = []
            for record in result:
                members = set(record['cluster_members'])
                if len(members) > 1:  # Only clusters with 2+ entities
                    clusters.append(members)
            
            print(f"✓ Found {len(clusters)} duplicate clusters")
            return clusters
    
    def merge_cluster(self, cluster: Set[str], canonical_id: str = None):
        """
        Merge all entities in a cluster into one
        
        Args:
            cluster: Set of entity IDs to merge
            canonical_id: Which entity to keep (default: first in set)
        """
        cluster_list = list(cluster)
        canonical_id = canonical_id or cluster_list[0]
        
        print(f"Merging cluster of {len(cluster)} entities into {canonical_id}")
        
        for entity_id in cluster_list:
            if entity_id != canonical_id:
                try:
                    self.merge_duplicate_entities(canonical_id, entity_id)
                except Exception as e:
                    print(f"Error merging {entity_id}: {e}")
    
    def get_statistics(self) -> Dict:
        """Get database statistics"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (e:Entity)
                RETURN 
                    count(e) AS total_entities,
                    count(CASE WHEN e.resolved = true THEN 1 END) AS resolved_entities,
                    count(CASE WHEN e.resolved = false OR e.resolved IS NULL THEN 1 END) AS unresolved_entities
            """)
            
            record = result.single()
            
            # Count similarity edges
            edge_result = session.run("""
                MATCH ()-[r:SIMILAR_TO]->()
                RETURN count(r) AS similarity_edges
            """)
            edge_record = edge_result.single()
            
            return {
                'total_entities': record['total_entities'],
                'resolved_entities': record['resolved_entities'],
                'unresolved_entities': record['unresolved_entities'],
                'similarity_edges': edge_record['similarity_edges']
            }
    
    def export_resolved_entities(self) -> List[Dict]:
        """Export all resolved entities"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (e:Entity)
                WHERE e.resolved = true
                RETURN e.id AS id,
                       e.name AS name,
                       e.type AS type,
                       e.merged_from AS merged_from
                ORDER BY e.name
            """)
            
            return [dict(record) for record in result]
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison"""
        import re
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()


class EntityResolutionPipeline:
    """Complete entity resolution pipeline"""
    
    def __init__(self, neo4j_uri: str, user: str, password: str):
        self.resolver = Neo4jEntityResolver(neo4j_uri, user, password)
    
    def run_pipeline(self, entities: List[Dict], 
                    auto_resolve_threshold: float = 0.92,
                    similarity_threshold: float = 0.85):
        """
        Run complete entity resolution pipeline
        
        Args:
            entities: List of entities to process
            auto_resolve_threshold: Threshold for automatic merging
            similarity_threshold: Threshold for similarity detection
        """
        print("\n" + "="*70)
        print("ENTITY RESOLUTION PIPELINE")
        print("="*70)
        
        # Step 1: Add entities
        print("\n### STEP 1: LOADING ENTITIES ###")
        self.resolver.add_entities(entities)
        
        # Step 2: Create similarity graph
        print("\n### STEP 2: CREATING SIMILARITY GRAPH ###")
        self.resolver.create_similarity_edges(similarity_threshold)
        
        # Step 3: Find duplicate candidates
        print("\n### STEP 3: FINDING DUPLICATES ###")
        candidates = self.resolver.find_duplicate_candidates(similarity_threshold)
        
        print("\nTop 10 duplicate candidates:")
        for i, candidate in enumerate(candidates[:10], 1):
            print(f"{i}. '{candidate['entity1_name']}' ~ '{candidate['entity2_name']}' "
                  f"(similarity: {candidate['similarity']:.3f})")
        
        # Step 4: Auto-resolve high-confidence duplicates
        print("\n### STEP 4: AUTO-RESOLVING HIGH-CONFIDENCE DUPLICATES ###")
        self.resolver.auto_resolve_duplicates(auto_resolve_threshold)
        
        # Step 5: Find clusters
        print("\n### STEP 5: FINDING DUPLICATE CLUSTERS ###")
        clusters = self.resolver.find_duplicate_clusters()
        
        if clusters:
            print(f"\nFound {len(clusters)} clusters:")
            for i, cluster in enumerate(clusters[:5], 1):
                print(f"  Cluster {i}: {len(cluster)} entities")
        
        # Step 6: Statistics
        print("\n### STEP 6: FINAL STATISTICS ###")
        stats = self.resolver.get_statistics()
        print(f"\n📊 Database Statistics:")
        print(f"  Total entities: {stats['total_entities']}")
        print(f"  Resolved (merged): {stats['resolved_entities']}")
        print(f"  Unresolved: {stats['unresolved_entities']}")
        print(f"  Similarity edges: {stats['similarity_edges']}")
        
        print("\n" + "="*70)
        print("✅ PIPELINE COMPLETE")
        print("="*70 + "\n")
    
    def close(self):
        self.resolver.close()


def example_usage():
    """Example usage with sample data"""
    
    # Sample entities with duplicates
    entities = [
        # Duplicate group 1: Elon Musk
        {'id': 'e1', 'name': 'Elon Musk', 'type': 'PERSON', 'properties': {'role': 'CEO'}},
        {'id': 'e2', 'name': 'Elon R. Musk', 'type': 'PERSON', 'properties': {'role': 'Founder'}},
        {'id': 'e3', 'name': 'Musk, Elon', 'type': 'PERSON', 'properties': {'company': 'Tesla'}},
        
        # Duplicate group 2: Tesla
        {'id': 'e4', 'name': 'Tesla', 'type': 'ORGANIZATION', 'properties': {'industry': 'automotive'}},
        {'id': 'e5', 'name': 'Tesla Inc.', 'type': 'ORGANIZATION', 'properties': {'founded': '2003'}},
        {'id': 'e6', 'name': 'Tesla Motors', 'type': 'ORGANIZATION', 'properties': {'location': 'Austin'}},
        
        # Duplicate group 3: Microsoft
        {'id': 'e7', 'name': 'Microsoft', 'type': 'ORGANIZATION', 'properties': {}},
        {'id': 'e8', 'name': 'Microsoft Corporation', 'type': 'ORGANIZATION', 'properties': {}},
        {'id': 'e9', 'name': 'Microsoft Corp.', 'type': 'ORGANIZATION', 'properties': {}},
        
        # Similar but NOT duplicates
        {'id': 'e10', 'name': 'Bill Gates', 'type': 'PERSON', 'properties': {}},
        {'id': 'e11', 'name': 'Jeff Bezos', 'type': 'PERSON', 'properties': {}},
        {'id': 'e12', 'name': 'Amazon', 'type': 'ORGANIZATION', 'properties': {}},
        {'id': 'e13', 'name': 'Apple', 'type': 'ORGANIZATION', 'properties': {}},
        
        # Edge cases
        {'id': 'e14', 'name': 'Tim Cook', 'type': 'PERSON', 'properties': {}},
        {'id': 'e15', 'name': 'Timothy Cook', 'type': 'PERSON', 'properties': {}},
    ]
    
    # Connection details (adjust for your Neo4j instance)
    NEO4J_URI = "neo4j+ssc://24c7b521.databases.neo4j.io:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]
    
    try:
        # Run pipeline
        pipeline = EntityResolutionPipeline(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        
        pipeline.run_pipeline(
            entities,
            auto_resolve_threshold=0.92,
            similarity_threshold=0.85
        )
        
        # Export results
        resolved = pipeline.resolver.export_resolved_entities()
        
        print("\n📝 Resolved Entities:")
        for entity in resolved[:10]:
            merged_from = entity.get('merged_from', [])
            if merged_from:
                print(f"  • {entity['name']} (merged from: {', '.join(merged_from)})")
        
        pipeline.close()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure:")
        print("  1. Neo4j is running")
        print("  2. APOC plugin is installed")
        print("  3. GDS library is installed")
        print("  4. Connection details are correct")


if __name__ == "__main__":
    example_usage()

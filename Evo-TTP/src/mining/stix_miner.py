"""
STIX Miner Module - Phase 1: Semantic Knowledge Mining

This module handles parsing MITRE ATT&CK STIX data and identifying
structural holes using semantic embeddings.

Author: Dilkhaz Mohammed
Institution: University of Mohaghegh Ardabili
"""

import logging
import networkx as nx
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from stix2 import MemoryStore, Filter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


@dataclass
class TTPNode:
    """Represents a node in the MITRE ATT&CK knowledge graph."""
    id: str
    name: str
    description: str
    vector: np.ndarray = None
    tactic: str = None
    platform: List[str] = None

    def __post_init__(self):
        if self.platform is None:
            self.platform = []


class STIXMiner:
    """
    Parses MITRE ATT&CK JSON into a High-Dimensional Vector Space.
    Detects 'Structural Holes' to seed new threat generation.

    Structural holes represent pairs of techniques that are semantically similar
    but lack a documented relationship in the MITRE framework.
    """

    def __init__(
        self,
        stix_path: Optional[str] = None,
        embedding_model: str = 'all-MiniLM-L6-v_threshold: float =2',
        similarity 0.75
    ):
        """
        Initialize the STIX Miner.

        Args:
            stix_path: Path to MITRE STIX JSON file
            embedding_model: Name of Sentence-BERT model to use
            similarity_threshold: Threshold for identifying structural holes
        """
        self.embedding_model_name = embedding_model
        self.similarity_threshold = similarity_threshold

        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedder = SentenceTransformer(embedding_model)

        # Initialize knowledge graph
        self.graph = nx.DiGraph()
        self.nodes: Dict[str, TTPNode] = {}

        # Load data
        if stix_path:
            logger.info(f"Loading MITRE data from: {stix_path}")
            self.store = MemoryStore()
            self.store.load_from_file(stix_path)
            self._parse_real_data()
        else:
            logger.warning("No MITRE file provided. Using MOCK data for demonstration.")
            self._generate_mock_data()

    def _generate_mock_data(self):
        """Generates fake STIX objects for code validation without downloading file."""
        mock_techniques = [
            ("T1574", "Hijack Execution Flow",
             "Adversaries execute malicious code by hijacking order of operations."),
            ("T1090", "Proxy",
             "Adversaries use connection proxy to hide identity."),
            ("T1609", "Container Administration Command",
             "Adversaries abuse Docker commands to escape."),
            ("T1070", "Indicator Removal",
             "Adversaries delete logs to hide tracks."),
            ("T1059", "Command and Scripting Interpreter",
             "Adversaries may abuse command and script interpreters to execute commands."),
            ("T1566", "Phishing",
             "Adversaries may send phishing messages to gain access to victim systems."),
        ]

        logger.info(f"Generating mock data for {len(mock_techniques)} techniques")

        for tid, name, desc in mock_techniques:
            vec = self.embedder.encode(desc)
            self.nodes[tid] = TTPNode(
                id=tid,
                name=name,
                description=desc,
                vector=vec
            )
            self.graph.add_node(tid)

    def _parse_real_data(self):
        """Parses actual MITRE STIX JSON objects."""
        logger.info("Parsing MITRE ATT&CK data...")

        # Extract Techniques
        techniques = self.store.query([Filter("type", "=", "attack-pattern")])
        logger.info(f"Found {len(techniques)} attack patterns")

        descriptions = []
        node_ids = []

        for t in techniques:
            node_ids.append(t.id)
            desc = t.description if hasattr(t, 'description') and t.description else t.name
            descriptions.append(desc)

        # Vectorize Descriptions (Latent Space)
        logger.info("Generating semantic embeddings...")
        vectors = self.embedder.encode(descriptions, show_progress_bar=True)

        for nid, desc, vec in zip(node_ids, descriptions, vectors):
            self.nodes[nid] = TTPNode(
                id=nid,
                name="Technique",
                description=desc,
                vector=vec
            )
            self.graph.add_node(nid)

        logger.info(f"Parsed {len(self.nodes)} techniques into semantic vectors")

    def find_structural_holes(self, threshold: float = None) -> List[Tuple[str, str]]:
        """
        Identify pairs of techniques that are Semantically Similar but NOT Connected.

        This represents the 'Hidden Pattern' or 'Structural Hole'.

        Args:
            threshold: Similarity threshold (uses default if None)

        Returns:
            List of tuples (source_technique_name, target_technique_name)
        """
        if threshold is None:
            threshold = self.similarity_threshold

        logger.info(f"[*] Mining Structural Holes in Graph (Threshold={threshold})...")

        node_keys = list(self.nodes.keys())
        vectors = np.stack([self.nodes[n].vector for n in node_keys])

        # Calculate Cosine Similarity Matrix
        logger.info("Computing similarity matrix...")
        sim_matrix = cosine_similarity(vectors)

        holes = []
        count = 0

        for i in range(len(node_keys)):
            for j in range(i + 1, len(node_keys)):
                # If vector similarity is HIGH but Graph Edge is MISSING
                if sim_matrix[i][j] > threshold:
                    source, target = node_keys[i], node_keys[j]

                    if not self.graph.has_edge(source, target):
                        holes.append((
                            self.nodes[source].name,
                            self.nodes[target].name
                        ))
                        count += 1

        logger.info(f"[*] Discovered {count} Structural Holes to bridge with Synthetic AI.")

        return holes

    def get_similarity_matrix(self) -> np.ndarray:
        """Get the cosine similarity matrix for all node pairs."""
        node_keys = list(self.nodes.keys())
        vectors = np.stack([self.nodes[n].vector for n in node_keys])
        return cosine_similarity(vectors)

    def export_graph(self, output_path: str):
        """Export the knowledge graph to a file."""
        nx.write_gexf(self.graph, output_path)
        logger.info(f"Graph exported to {output_path}")

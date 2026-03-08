#!/usr/bin/env python3
"""
Evo-TTP: Generative and Robust Prediction of Novel Cyber Threat Tactics
using Adversarial Fine-Tuning of Large Language Models

Main Pipeline Orchestrator
=========================
This script orchestrates the three-phase Evo-TTP pipeline:
1. Semantic Knowledge Mining (STIX-Miner)
2. Synthetic Data Expansion (Teacher-Student)
3. Adversarial GRPO Training

Author: Dilkhaz Mohammed, Shahram Jamali
Institution: University of Mohaghegh Ardabili
License: MIT
"""

import json
import torch
import random
import logging
import argparse
import networkx as nx
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime

# Big Data Libraries
from stix2 import MemoryStore, Filter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Generative AI Training Libraries
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

# Import local modules
from src.mining.stix_miner import STIXMiner
from src.synthesis.teacher import SyntheticTeacher
from src.training.grpo_trainer import EvoTTPTrainer
from config.settings import Settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==============================================================================
# Configuration Constants
# ==============================================================================
@dataclass
class TTPNode:
    """Represents a node in the MITRE ATT&CK knowledge graph."""
    id: str
    name: str
    description: str
    vector: np.ndarray = None
    tactic: str = None
    platform: List[str] = field(default_factory=list)


# ==============================================================================
# Main Pipeline Class
# ==============================================================================
class EvoTTPipeline:
    """
    Orchestrates the complete Evo-TTP pipeline for predicting novel cyber threat tactics.

    The pipeline consists of three phases:
    1. Semantic Mining: Extracts structural holes from MITRE ATT&CK
    2. Synthesis: Generates synthetic training data via teacher-student architecture
    3. Training: Fine-tunes student model using adversarial GRPO
    """

    def __init__(self, config: Settings):
        """
        Initialize the Evo-TTP pipeline with configuration.

        Args:
            config: Configuration object containing all hyperparameters
        """
        self.config = config
        self.miner: Optional[STIXMiner] = None
        self.teacher: Optional[SyntheticTeacher] = None
        self.trainer: Optional[EvoTTPTrainer] = None
        self.structural_holes: List[Tuple[str, str]] = []
        self.synthetic_dataset: List[Dict] = []

        # Set random seeds for reproducibility
        self._set_seeds()

    def _set_seeds(self):
        """Set random seeds for reproducibility across all libraries."""
        torch.manual_seed(self.config.SEED)
        torch.cuda.manual_seed_all(self.config.SEED)
        np.random.seed(self.config.SEED)
        random.seed(self.config.SEED)
        logger.info(f"Random seeds set to {self.config.SEED} for reproducibility")

    def run_phase1_mining(self, stix_path: Optional[str] = None) -> List[Tuple[str, str]]:
        """
        Phase 1: Semantic Knowledge Mining

        Parses MITRE ATT&CK data and identifies structural holes -
        pairs of techniques that are semantically similar but not connected.

        Args:
            stix_path: Path to STIX JSON file (optional, uses mock data if None)

        Returns:
            List of tuples representing structural holes (source_technique, target_technique)
        """
        logger.info("=" * 60)
        logger.info("PHASE 1: Semantic Knowledge Mining")
        logger.info("=" * 60)

        # Initialize STIX Miner
        self.miner = STIXMiner(
            stix_path=stix_path,
            embedding_model=self.config.EMBEDDING_MODEL,
            similarity_threshold=self.config.SIMILARITY_THRESHOLD
        )

        # Find structural holes
        self.structural_holes = self.miner.find_structural_holes(
            threshold=self.config.SIMILARITY_THRESHOLD
        )

        logger.info(f"Phase 1 Complete: Discovered {len(self.structural_holes)} structural holes")

        # Save results
        self._save_structural_holes()

        return self.structural_holes

    def run_phase2_synthesis(self) -> List[Dict]:
        """
        Phase 2: Synthetic Data Expansion

        Uses teacher-student architecture to generate synthetic training data
        that bridges identified structural holes.

        Returns:
            List of dictionaries containing prompt-response pairs
        """
        logger.info("=" * 60)
        logger.info("PHASE 2: Synthetic Data Expansion")
        logger.info("=" * 60)

        if not self.structural_holes:
            logger.warning("No structural holes found. Run Phase 1 first.")
            return []

        # Initialize Teacher Model
        self.teacher = SyntheticTeacher(
            model_name=self.config.TEACHER_MODEL,
            temperature=self.config.TEACHER_TEMPERATURE,
            max_tokens=self.config.MAX_NEW_TOKENS
        )

        # Generate synthetic dataset
        self.synthetic_dataset = self.teacher.generate_dataset(
            holes=self.structural_holes,
            expansion_factor=self.config.EXPANSION_FACTOR
        )

        logger.info(f"Phase 2 Complete: Generated {len(self.synthetic_dataset)} synthetic samples")

        # Save dataset
        self._save_synthetic_dataset()

        return self.synthetic_dataset

    def run_phase3_training(self, dataset_path: Optional[str] = None):
        """
        Phase 3: Adversarial GRPO Training

        Fine-tunes the student model using Group Relative Policy Optimization
        with adversarial robustness constraints.

        Args:
            dataset_path: Path to synthetic dataset (uses self.synthetic_dataset if None)
        """
        logger.info("=" * 60)
        logger.info("PHASE 3: Adversarial GRPO Training")
        logger.info("=" * 60)

        # Check for GPU availability
        if not torch.cuda.is_available():
            logger.error("CUDA is required for training. No GPU detected.")
            return

        # Load or use existing dataset
        if dataset_path:
            with open(dataset_path, 'r') as f:
                dataset = json.load(f)
        else:
            dataset = self.synthetic_dataset

        if not dataset:
            logger.warning("No training data available. Run Phase 2 first.")
            return

        # Initialize Trainer
        self.trainer = EvoTTPTrainer(
            model_name=self.config.STUDENT_MODEL,
            max_seq_length=self.config.MAX_SEQ_LENGTH,
            lora_r=self.config.LORA_R,
            lora_alpha=self.config.LORA_ALPHA,
            learning_rate=self.config.LEARNING_RATE,
            batch_size=self.config.BATCH_SIZE,
            gradient_steps=self.config.GRADIENT_ACCUMULATION_STEPS,
            reward_weights=self.config.REWARD_WEIGHTS
        )

        # Run training
        self.trainer.train(
            dataset=dataset,
            num_training_steps=self.config.NUM_TRAINING_STEPS,
            checkpoint_interval=self.config.CHECKPOINT_INTERVAL
        )

        # Save final model
        self._save_trained_model()

        logger.info("Phase 3 Complete: Model training finished")

    def run_full_pipeline(self, stix_path: Optional[str] = None):
        """
        Execute the complete three-phase pipeline.

        Args:
            stix_path: Path to MITRE STIX JSON file
        """
        logger.info("Starting Evo-TTP Full Pipeline")
        logger.info(f"Configuration: {self.config}")

        start_time = datetime.now()

        try:
            # Phase 1: Mining
            self.run_phase1_mining(stix_path)

            # Phase 2: Synthesis
            self.run_phase2_synthesis()

            # Phase 3: Training
            self.run_phase3_training()

            end_time = datetime.now()
            duration = end_time - start_time

            logger.info("=" * 60)
            logger.info("PIPELINE COMPLETE")
            logger.info(f"Total Duration: {duration}")
            logger.info("=" * 60)

        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise

    def _save_structural_holes(self):
        """Save discovered structural holes to JSON file."""
        output_path = self.config.OUTPUT_DIR / "structural_holes.json"

        holes_data = [
            {"source": src, "target": tgt}
            for src, tgt in self.structural_holes
        ]

        with open(output_path, 'w') as f:
            json.dump(holes_data, f, indent=2)

        logger.info(f"Structural holes saved to {output_path}")

    def _save_synthetic_dataset(self):
        """Save synthetic dataset to JSON file."""
        output_path = self.config.OUTPUT_DIR / "synthetic_dataset.json"

        with open(output_path, 'w') as f:
            json.dump(self.synthetic_dataset, f, indent=2)

        logger.info(f"Synthetic dataset saved to {output_path}")

    def _save_trained_model(self):
        """Save trained model and tokenizer."""
        if self.trainer:
            output_path = self.config.OUTPUT_DIR / "final_model"
            self.trainer.save_model(str(output_path))
            logger.info(f"Trained model saved to {output_path}")


# ==============================================================================
# Command Line Interface
# ==============================================================================
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evo-TTP: Generative and Robust Prediction of Novel Cyber Threat Tactics"
    )

    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2, 3, 99],
        default=99,
        help="Phase to run: 1=Mining, 2=Synthesis, 3=Training, 99=Full Pipeline"
    )

    parser.add_argument(
        "--stix-path",
        type=str,
        default=None,
        help="Path to MITRE ATT&CK STIX JSON file"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/default_config.json",
        help="Path to configuration file"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory for results and models"
    )

    return parser.parse_args()


def main():
    """Main entry point for the Evo-TTP pipeline."""
    args = parse_args()

    # Load configuration
    config = Settings.from_file(args.config) if Path(args.config).exists() else Settings()
    config.OUTPUT_DIR = Path(args.output_dir)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize pipeline
    pipeline = EvoTTPipeline(config)

    # Run requested phase(s)
    if args.phase == 1:
        pipeline.run_phase1_mining(args.stix_path)
    elif args.phase == 2:
        pipeline.run_phase2_synthesis()
    elif args.phase == 3:
        pipeline.run_phase3_training()
    elif args.phase == 99:
        pipeline.run_full_pipeline(args.stix_path)


if __name__ == "__main__":
    main()

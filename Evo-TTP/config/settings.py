"""
Evo-TTP Configuration Settings

This file contains all configurable parameters for the Evo-TTP pipeline.
Can be loaded from JSON or modified directly.

Author: Dilkhaz Mohammed
Institution: University of Mohaghegh Ardabili
"""

import json
import os
from pathlib import Path
from typing import Tuple, Optional
from dataclasses import dataclass, asdict


@dataclass
class Settings:
    """Main configuration class for Evo-TTP."""

    # ========================================================================
    # Data Source Settings
    # ========================================================================
    MITRE_DATA_URL: str = "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json"
    MITRE_VERSION: str = "v18.0"
    MITRE_COMMIT_HASH: str = "a28c3d8"  # Exact commit for reproducibility

    # ========================================================================
    # Phase 1: Mining Settings
    # ========================================================================
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    SIMILARITY_THRESHOLD: float = 0.75
    GRID_SEARCH_THRESHOLDS: Tuple[float, ...] = (0.70, 0.75, 0.80, 0.85)

    # ========================================================================
    # Phase 2: Synthesis Settings
    # ========================================================================
    TEACHER_MODEL: str = "meta-llama/Llama-3.1-405B-Instruct"
    TEACHER_TEMPERATURE: float = 0.7
    TEACHER_TOP_P: float = 0.9
    MAX_NEW_TOKENS: int = 1024
    REPETITION_PENALTY: float = 1.1
    EXPANSION_FACTOR: int = 44  # Variants per structural hole

    # Feasibility filter settings
    FEASIBILITY_MODEL: str = "microsoft/securitybert-base"
    FEASIBILITY_THRESHOLD: float = 0.6

    # ========================================================================
    # Phase 3: Training Settings
    # ========================================================================
    STUDENT_MODEL: str = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
    MAX_SEQ_LENGTH: int = 2048

    # LoRA settings
    LORA_R: int = 64
    LORA_ALPHA: int = 128
    LORA_DROPOUT: float = 0.05

    # Training hyperparameters
    LEARNING_RATE: float = 5e-6
    BATCH_SIZE: int = 4
    GRADIENT_ACCUMULATION_STEPS: int = 4
    NUM_TRAINING_STEPS: int = 12000
    WARMUP_STEPS: int = 500

    # GRPO settings
    GROUP_SIZE: int = 8
    # (novelty, feasibility, impact, brittleness_penalty)
    REWARD_WEIGHTS: Tuple[float, float, float, float] = (0.4, 0.3, 0.2, 0.5)

    # Adversarial settings
    PERTURBATION_PROB: float = 0.3

    # ========================================================================
    # Evaluation Settings
    # ========================================================================
    TEST_SET_SIZE: int = 500
    NUM_GENERATIONS_PER_PROMPT: int = 5

    # ========================================================================
    # System Settings
    # ========================================================================
    SEED: int = 42
    OUTPUT_DIR: Path = Path("output")
    CHECKPOINT_INTERVAL: int = 1000

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'Settings':
        """Create Settings from dictionary."""
        return cls(**config_dict)

    @classmethod
    def from_file(cls, file_path: str) -> 'Settings':
        """Load Settings from JSON file."""
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def to_dict(self) -> dict:
        """Convert Settings to dictionary."""
        config_dict = asdict(self)
        # Handle Path serialization
        config_dict['OUTPUT_DIR'] = str(self.OUTPUT_DIR)
        config_dict['GRID_SEARCH_THRESHOLDS'] = list(self.GRID_SEARCH_THRESHOLDS)
        config_dict['REWARD_WEIGHTS'] = list(self.REWARD_WEIGHTS)
        return config_dict

    def save(self, file_path: str):
        """Save Settings to JSON file."""
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def __str__(self) -> str:
        """String representation of settings."""
        lines = ["Evo-TTP Configuration:", "=" * 40]

        # Data Source
        lines.append("\n[Data Source]")
        lines.append(f"  MITRE Version: {self.MITRE_VERSION}")
        lines.append(f"  Commit Hash: {self.MITRE_COMMIT_HASH}")

        # Mining
        lines.append("\n[Mining]")
        lines.append(f"  Embedding Model: {self.EMBEDDING_MODEL}")
        lines.append(f"  Similarity Threshold: {self.SIMILARITY_THRESHOLD}")

        # Synthesis
        lines.append("\n[Synthesis]")
        lines.append(f"  Teacher Model: {self.TEACHER_MODEL}")
        lines.append(f"  Expansion Factor: {self.EXPANSION_FACTOR}")

        # Training
        lines.append("\n[Training]")
        lines.append(f"  Student Model: {self.STUDENT_MODEL}")
        lines.append(f"  Learning Rate: {self.LEARNING_RATE}")
        lines.append(f"  LoRA Rank: {self.LORA_R}")
        lines.append(f"  Training Steps: {self.NUM_TRAINING_STEPS}")

        # Reward Weights
        lines.append("\n[Reward Weights]")
        lines.append(f"  Novelty (α): {self.REWARD_WEIGHTS[0]}")
        lines.append(f"  Feasibility (β): {self.REWARD_WEIGHTS[1]}")
        lines.append(f"  Impact (γ): {self.REWARD_WEIGHTS[2]}")
        lines.append(f"  Brittleness Penalty (λ): {self.REWARD_WEIGHTS[3]}")

        return "\n".join(lines)


# ==============================================================================
# Default Configuration
# ==============================================================================
DEFAULT_CONFIG = Settings()

# Environment variable overrides
def load_config_from_env() -> Settings:
    """Load configuration with environment variable overrides."""
    config = DEFAULT_CONFIG

    # Override with environment variables if present
    if os.getenv("EVO_TTP_SEED"):
        config.SEED = int(os.getenv("EVO_TTP_SEED"))
    if os.getenv("EVO_TTP_OUTPUT_DIR"):
        config.OUTPUT_DIR = Path(os.getenv("EVO_TTP_OUTPUT_DIR"))
    if os.getenv("EVO_TTP_LORA_R"):
        config.LORA_R = int(os.getenv("EVO_TTP_LORA_R"))

    return config


if __name__ == "__main__":
    # Print default configuration
    print(DEFAULT_CONFIG)

    # Save to file
    DEFAULT_CONFIG.save("config/default_config.json")
    print("\nConfiguration saved to config/default_config.json")

"""
Evo-TTP Source Package

Author: Dilkhaz Mohammed
Institution: University of Mohaghegh Ardabili
"""

__version__ = "1.0.0"
__author__ = "Dilkhaz Mohammed, Shahram Jamali"

from .mining.stix_miner import STIXMiner, TTPNode
from .synthesis.teacher import SyntheticTeacher, FeasibilityFilter
from .training.grpo_trainer import EvoTTPTrainer, GRPOConfig
from .evaluation.evaluate import EvaluationMetrics, AdvBenchCTI

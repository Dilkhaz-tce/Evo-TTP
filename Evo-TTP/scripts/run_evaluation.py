#!/usr/bin/env python3
"""
Evaluation Script for Evo-TTP

This script runs comprehensive evaluation on the trained model,
computing all metrics: novelty, feasibility, impact, and robustness.

Author: Dilkhaz Mohammed
Institution: University of Mohaghegh Ardabili
"""

import json
import logging
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List

import torch
from sentence_transformers import SentenceTransformer

from src.evaluation.evaluate import (
    EvaluationMetrics,
    AdvBenchCTI,
    run_evaluation
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model(model_path: str):
    """Load the trained model."""
    try:
        from unsloth import FastLanguageModel

        logger.info(f"Loading model from {model_path}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_path,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True
        )

        # Set to evaluation mode
        model.eval()

        return model, tokenizer
    except Exception as e:
        logger.warning(f"Could not load model: {e}")
        logger.warning("Running evaluation in mock mode")
        return None, None


def load_dataset(dataset_path: str) -> List[Dict]:
    """Load evaluation dataset."""
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    return dataset


def create_perturbation_fn():
    """Create adversarial perturbation function."""
    def perturbation(text: str) -> str:
        # Simple character swap perturbation
        chars = list(text)
        if len(chars) > 5:
            idx = len(chars) // 2
            if idx > 0:
                chars[idx], chars[idx - 1] = chars[idx - 1], chars[idx]
        return "".join(chars)

    return perturbation


def main():
    """Main evaluation entry point."""
    parser = argparse.ArgumentParser(description="Evo-TTP Evaluation")

    parser.add_argument(
        "--model-path",
        type=str,
        default="output/final_model",
        help="Path to trained model"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="data/test_set.json",
        help="Path to evaluation dataset"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="output/evaluation_results.json",
        help="Output path for results"
    )

    parser.add_argument(
        "--mitre-embeddings",
        type=str,
        default=None,
        help="Path to pre-computed MITRE embeddings"
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Evo-TTP Evaluation")
    logger.info("=" * 60)

    # Load model (if available)
    model, tokenizer = load_model(args.model_path)

    # Load dataset
    logger.info(f"Loading dataset from {args.dataset}")
    dataset = load_dataset(args.dataset)
    logger.info(f"Loaded {len(dataset)} evaluation samples")

    # Initialize evaluator
    evaluator = EvaluationMetrics(
        embedding_model="all-MiniLM-L6-v2",
        mitre_embeddings_path=args.mitre_embeddings
    )

    # Extract generated TTPs
    generated_ttps = [item.get("response", "") for item in dataset]
    prompts = [item.get("prompt", "") for item in dataset]

    # Compute metrics
    logger.info("Computing novelty score...")
    novelty = evaluator.compute_novelty(generated_ttps)
    logger.info(f"Novelty: {novelty:.4f}")

    logger.info("Computing feasibility score...")
    feasibility = evaluator.compute_feasibility(generated_ttps)
    logger.info(f"Feasibility: {feasibility:.4f}")

    logger.info("Computing impact score...")
    impact = evaluator.compute_impact(generated_ttps)
    logger.info(f"Impact: {impact:.4f}")

    # Aggregate utility
    aggregate_utility = 0.4 * novelty + 0.3 * feasibility + 0.3 * impact
    logger.info(f"Aggregate Utility: {aggregate_utility:.4f}")

    # Compute robustness (if model available)
    robustness_results = {}
    if model is not None and tokenizer is not None:
        logger.info("Computing robustness metrics...")

        perturbation_fn = create_perturbation_fn()

        asr, consistency = evaluator.compute_robustness(
            model, tokenizer, prompts[:10], perturbation_fn
        )

        robustness_results = {
            "attack_success_rate": asr,
            "semantic_consistency": consistency
        }

        logger.info(f"Attack Success Rate: {asr:.4f}")
        logger.info(f"Semantic Consistency: {consistency:.4f}")

    # Compile results
    results = {
        "novelty_score": novelty,
        "feasibility_score": feasibility,
        "impact_score": impact,
        "aggregate_utility": aggregate_utility,
        **robustness_results,
        "num_samples": len(generated_ttps)
    }

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info("=" * 60)
    logger.info("Evaluation Complete!")
    logger.info(f"Results saved to {output_path}")
    logger.info("=" * 60)

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Novelty Score:        {novelty:.4f}")
    print(f"  Feasibility Score:    {feasibility:.4f}")
    print(f"  Impact Score:         {impact:.4f}")
    print(f"  Aggregate Utility:    {aggregate_utility:.4f}")
    if robustness_results:
        print(f"  Attack Success Rate:  {robustness_results['attack_success_rate']:.4f}")
        print(f"  Semantic Consistency: {robustness_results['semantic_consistency']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()

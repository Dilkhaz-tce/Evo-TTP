# Evo-TTP: Generative and Robust Prediction of Novel Cyber Threat Tactics

<p align="center">
  <img src="docs/evo_ttp_logo.png" alt="Evo-TTP Logo" width="400"/>
</p>

<p align="center">
  <a href="https://github.com/Evo-TTP/Evo-TTP/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  </a>
  <a href="https://github.com/Evo-TTP/Evo-TTP/releases">
    <img src="https://img.shields.io/badge/Version-1.0.0-blue.svg" alt="Version">
  </a>
  <a href="https://arxiv.org">
    <img src="https://img.shields.io/badge/Paper-ArXiv-orange.svg" alt="ArXiv">
  </a>
</p>

## Overview

Evo-TTP is a comprehensive framework for the predictive generation of novel and robust tactics, techniques, and procedures (TTPs) using big data mining and adversarial learning. By leveraging the MITRE ATT&CK® framework and large language models, Evo-TTP addresses the critical challenge of predicting future cyber threats before they materialize.

### Key Features

- **Structural Hole Analysis**: Identifies hidden attack patterns in the MITRE ATT&CK knowledge graph
- **Synthetic Data Generation**: Uses teacher-student architecture (Llama-405B → Llama-8B) to overcome data scarcity
- **Adversarial GRPO Training**: Implements Group Relative Policy Optimization with robustness constraints
- **Multi-dimensional Evaluation**: Comprehensive metrics for novelty, feasibility, impact, and robustness

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Evo-TTP Pipeline                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Phase 1: Semantic Mining         Phase 2: Synthesis            │
│  ┌──────────────────┐           ┌──────────────────┐          │
│  │ MITRE ATT&CK    │           │ Structural Holes  │          │
│  │ v18.0 JSON      │──────────▶│ (Target Prompts) │          │
│  └────────┬─────────┘           └────────┬─────────┘          │
│           │                             │                      │
│           ▼                             ▼                      │
│  ┌──────────────────┐           ┌──────────────────┐          │
│  │ STIX-Miner      │           │ Teacher Model    │          │
│  │ + Sentence-BERT │           │ (Llama-405B)     │          │
│  └────────┬─────────┘           └────────┬─────────┘          │
│           │                             │                      │
│           ▼                             ▼                      │
│  ┌──────────────────┐           ┌──────────────────┐          │
│  │ Semantic Graph   │           │ Synthetic Data   │          │
│  │ + Holes (342)   │──────────▶│ (15,200 samples) │          │
│  └──────────────────┘           └──────────────────┘          │
│                                                                  │
│  Phase 3: Training                                               │
│  ┌──────────────────────────────────────────────────────┐       │
│  │              Adversarial GRPO                        │       │
│  │  ┌─────────┐  ┌──────────┐  ┌─────────────────┐   │       │
│  │  │Student  │─▶│ GRPO     │─▶│ Fine-tuned      │   │       │
│  │  │(Llama-8B)│  │Training  │  │ Evo-TTP Model  │   │       │
│  │  └─────────┘  └──────────┘  └─────────────────┘   │       │
│  │       ▲              │                               │       │
│  │       │              ▼                               │       │
│  │  ┌─────────────────────────────────────┐           │       │
│  │  │  Composite Reward:                    │           │       │
│  │  │  R = αR_novelty + βR_feasible        │           │       │
│  │  │      + γR_impact - λP_brittle        │           │       │
│  │  └─────────────────────────────────────┘           │       │
│  └──────────────────────────────────────────────────────┘       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites

- Python 3.11+
- CUDA-capable GPU (for training)
- 80GB+ GPU memory (recommended for full training)

### Clone Repository

```bash
git clone https://github.com/Evo-TTP/Evo-TTP.git
cd Evo-TTP
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Download MITRE ATT&CK Data

```bash
# Download MITRE ATT&CK Enterprise Matrix
wget https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json
```

### 2. Run Full Pipeline

```bash
python evo_ttp_main.py --phase 99 --stix-path enterprise-attack.json
```

### 3. Run Individual Phases

```bash
# Phase 1: Semantic Mining
python evo_ttp_main.py --phase 1 --stix-path enterprise-attack.json

# Phase 2: Synthesis
python evo_ttp_main.py --phase 2

# Phase 3: Training
python evo_ttp_main.py --phase 3
```

## Configuration

Edit `config/default_config.json` to customize parameters:

```json
{
  "SIMILARITY_THRESHOLD": 0.75,
  "TEACHER_MODEL": "meta-llama/Llama-3.1-405B-Instruct",
  "STUDENT_MODEL": "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
  "LORA_R": 64,
  "LEARNING_RATE": 5e-6,
  "REWARD_WEIGHTS": [0.4, 0.3, 0.2, 0.5]
}
```

## Project Structure

```
Evo-TTP/
├── evo_ttp_main.py           # Main pipeline orchestrator
├── config/
│   └── settings.py           # Configuration management
├── src/
│   ├── mining/
│   │   └── stix_miner.py    # Phase 1: Semantic mining
│   ├── synthesis/
│   │   └── teacher.py       # Phase 2: Synthetic generation
│   ├── training/
│   │   └── grpo_trainer.py  # Phase 3: Adversarial training
│   └── evaluation/
│       └── evaluate.py       # Evaluation metrics
├── scripts/
│   └── run_evaluation.py    # Evaluation script
├── data/                     # Data directory
├── docs/                     # Documentation
├── output/                   # Generated outputs
└── requirements.txt         # Dependencies
```

## Results

| Metric | Base Model | Standard SFT | Evo-TTP (Ours) |
|--------|-----------|--------------|-----------------|
| Novelty Score | 0.42 | 0.35 | **0.81** |
| Feasibility Score | 0.60 | 0.78 | 0.72 |
| Impact Score | 0.45 | 0.55 | **0.79** |
| Aggregate Utility | 0.48 | 0.54 | **0.77** |
| Robustness (ASR) | N/A | 54.30% | **22.10%** |

## Evaluation

Run comprehensive evaluation:

```bash
python scripts/run_evaluation.py \
    --model-path output/final_model \
    --dataset data/test_set.json \
    --output results.json
```

## Reproducibility

This implementation ensures reproducibility through:

1. **Exact Data Version**: MITRE CTI commit hash (a28c3d8, January 15, 2025)
2. **Random Seeds**: All random seeds set to 42
3. **Deterministic Settings**: CUDA deterministic algorithms enabled
4. **Complete Hyperparameter Documentation**: All settings in config/

## License

MIT License - See [LICENSE](LICENSE) for details.

## Citation

If you use Evo-TTP in your research, please cite:

```bibtex
@article{evo-ttp-2026,
  title={Evo-TTP: Generative and Robust Prediction of Novel Cyber Threat Tactics using Adversarial Fine-Tuning of Large Language Models},
  author={Mohammed, Dilkhaz and Jamali, Shahram},
  journal={arXiv preprint},
  year={2026}
}
```

## Acknowledgments

- MITRE Corporation for the ATT&CK framework
- Unsloth AI for optimization libraries
- Meta AI for Llama models

## Contact

- Email: dilkhaz.mohammed@uma.ac.ir
- GitHub Issues: https://github.com/Dilkhaz-tce/Evo-TTP/issues

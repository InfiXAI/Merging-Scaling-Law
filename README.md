# Model Merging Scaling Laws in Large Language Models

This repository contains the official implementation for the paper **"Model Merging Scaling Laws in Large Language Models"**.

> **Abstract**: We study empirical scaling laws for language model merging measured by cross-entropy. Despite the wide use of merging in practice, there is no quantitative rule that predicts what happens as we merge more experts or scale the base model. We show that the expected cross-entropy under equal-weight merging follows a simple, two-factor law: at a fixed base size, the loss decreases roughly in proportion to the inverse of the number of merged experts, and when varying the base size, the loss decreases according to the same power-law trend observed in pretraining. The law has two interpretable components: a baseline floor determined by the base model and its shared direction across domains, and a merging gain that reflects how dispersed the task vectors are under the model's curvature. Although individual merge outcomes can fluctuate unpredictably, the average over random subsets reveals a clear inverse relationship with the number of experts, and the variability shrinks as more experts are included. This law turns trial-and-error into predictive planning for budget-aware expert counts, saturation detection, and rule comparison across model scales.

## Repository Structure

```
.
├── src/                    # Core source code
│   ├── expert_training/   # Code for expert model training
│   └── merge/             # Code for model merging algorithms
├── scripts/               # Execution scripts (SLURM-based)
│   ├── train.sh          # Example training script
│   ├── merge.sh          # Example model merging script
│   ├── eval.sh           # Example evaluation script
│   └── ...               # Additional dependency files
├── data/                  # Example data files
└── ENV.txt               # Python environment configuration
```

## Installation and Setup

### 1. Python Environment

Create a conda environment using the provided specification:

```bash
conda create --name merging-laws --file ENV.txt
conda activate merging-laws
```

### 2. Additional Dependencies

Our expert training implementation builds upon the **OpenRLHF** framework. After setting up the base environment, install OpenRLHF:

```bash
pip install openrlhf
```

*Note: Please refer to the official [OpenRLHF repository](https://github.com/OpenRLHF/OpenRLHF) for any additional installation requirements or troubleshooting.*

## Usage

### Expert Model Training

To train expert models on specialized domains:

```bash
cd scripts/
sbatch train.sh
```

Key configuration parameters in `train.sh`.

### Model Merging

Execute model merging with equal-weight averaging:

```bash
sbatch merge.sh
```

Configuration parameters in `merge.sh`.

### Evaluation

Evaluate merged models on cross-entropy metrics:

```bash
sbatch eval.sh
```

## Data Format

The `data/` directory contains example datasets in the expected format. 


# Adaptive Scale-Aware Panoptic Segmentation with Boundary Refinement

A novel panoptic segmentation approach that combines adaptive scale-aware feature fusion with boundary refinement for improved multi-scale object segmentation on COCO 2017. This project addresses the challenge of accurately segmenting objects at vastly different scales within the same scene through dynamic feature routing and learned distance transforms.

## Installation

```bash
pip install -e .
```

Or install dependencies directly:

```bash
pip install -r requirements.txt
```

## Quick Start

### Training

Train the model with default configuration:

```bash
python scripts/train.py
```

Train with ablation configuration (baseline without novel components):

```bash
python scripts/train.py --config configs/ablation.yaml
```

Resume training from checkpoint:

```bash
python scripts/train.py --resume checkpoints/best_model.pth
```

### Evaluation

Evaluate a trained model:

```bash
python scripts/evaluate.py --checkpoint checkpoints/best_model.pth
```

Generate visualizations:

```bash
python scripts/evaluate.py --checkpoint checkpoints/best_model.pth --visualize
```

### Prediction

Run inference on a single image:

```bash
python scripts/predict.py --image path/to/image.jpg --checkpoint checkpoints/best_model.pth
```

## Methodology

This project addresses the fundamental challenge in panoptic segmentation: accurately segmenting objects at vastly different scales within the same scene. Traditional approaches use fixed feature pyramid fusion strategies that treat all scales equally, leading to suboptimal performance when scenes contain both large background regions and small foreground objects.

Our approach introduces two key innovations:

### 1. Adaptive Scale-Aware Fusion

Instead of uniformly fusing features from all FPN levels, we use multi-head self-attention to dynamically weight features based on the input scene characteristics. The module:
- Applies spatial pooling to each FPN level to create scale-specific embeddings
- Uses multi-head attention to compute adaptive fusion weights based on object size distribution
- Performs scale-specific convolutions before weighted fusion
- Enables the model to focus computational resources on relevant scales for each input

This is fundamentally different from standard FPN fusion (simple concatenation or addition) as it learns to route information based on content.

### 2. Boundary Refinement with Learned Distance Transforms

Object boundaries are critical for panoptic segmentation quality but are often blurred by standard convolutions. Our boundary refinement module:
- Predicts continuous distance transform bins (not just binary boundaries)
- Uses learned distance representations to guide feature refinement
- Applies boundary-aware attention to FPN features before final prediction
- Provides explicit supervision for boundary quality through dedicated loss terms

Unlike post-processing boundary refinement (e.g., CRF), our approach integrates boundary awareness into the feature learning process itself.

## Architecture

The complete model pipeline:

1. **ResNet-50 backbone** extracts hierarchical features at multiple resolutions
2. **Feature Pyramid Network (FPN)** creates multi-scale feature representations (C2-C5)
3. **Adaptive Scale Fusion Module** (novel) dynamically fuses FPN levels using attention
4. **Boundary Refinement Module** (novel) predicts distance transforms and refines features
5. **Dual segmentation heads** produce semantic and instance predictions
6. **Combined loss function** with semantic, instance, boundary, and panoptic quality terms

## Results

| Metric | Target | Status |
|--------|--------|--------|
| Panoptic Quality (PQ) | 42.0 | Run training to reproduce |
| Segmentation Quality (SQ) | 78.0 | Run training to reproduce |
| Recognition Quality (RQ) | 52.0 | Run training to reproduce |
| Boundary IoU | 65.0 | Run training to reproduce |

To reproduce results, run:

```bash
python scripts/train.py
python scripts/evaluate.py --checkpoint checkpoints/best_model.pth
```

## Ablation Study

The project includes an ablation configuration that disables the novel components:

```bash
# Train baseline (no adaptive fusion, no boundary refinement)
python scripts/train.py --config configs/ablation.yaml

# Train full model (with novel components)
python scripts/train.py --config configs/default.yaml

# Compare results
python scripts/evaluate.py --checkpoint checkpoints/best_model.pth
```

## Configuration

All hyperparameters are configurable via YAML files in `configs/`:

- `configs/default.yaml` - Full model with all novel components
- `configs/ablation.yaml` - Baseline model without novel components

Key parameters:
- `model.use_boundary_refinement` - Enable/disable boundary refinement module
- `model.use_scale_adaptive_fusion` - Enable/disable adaptive scale fusion
- `training.learning_rate` - Initial learning rate
- `training.scheduler` - Learning rate scheduler (cosine, step, plateau)
- `loss.boundary_weight` - Weight for boundary loss term

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

Run tests with coverage:

```bash
pytest tests/ --cov=src --cov-report=term-missing
```

## Project Structure

```
adaptive-scale-aware-panoptic-segmentation-with-boundary-refinement/
├── src/
│   └── adaptive_scale_aware_panoptic_segmentation_with_boundary_refinement/
│       ├── data/              # Data loading and preprocessing
│       ├── models/            # Model architecture and components
│       ├── training/          # Training loop with LR scheduling
│       ├── evaluation/        # Metrics and analysis
│       └── utils/             # Configuration and utilities
├── tests/                     # Unit tests
├── configs/                   # YAML configuration files
├── scripts/                   # Training, evaluation, and prediction scripts
└── requirements.txt           # Project dependencies
```

## Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- CUDA-capable GPU recommended
- 16GB+ RAM for COCO dataset

## License

MIT License - Copyright (c) 2026 Alireza Shojaei. See [LICENSE](LICENSE) for details.

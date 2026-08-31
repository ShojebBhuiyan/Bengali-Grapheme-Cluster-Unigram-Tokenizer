# Bengali Grapheme-Cluster Unigram Tokenizer

Research implementation of a grapheme-cluster (akshara) initialized unigram tokenizer for Bengali, with a 2×2 ablation study and multi-metric evaluation.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
pip install -e .
# Optional GPU acceleration for evaluation/EDA:
pip install -e ".[gpu]"
# Or with CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu124
```

## Pipeline

```bash
python -m tokenizer_bn.run_pipeline build-corpus
python -m tokenizer_bn.run_pipeline eda
python -m tokenizer_bn.run_pipeline train
python -m tokenizer_bn.run_pipeline evaluate
python -m tokenizer_bn.run_pipeline all
```

Use `--force` to re-run a completed step. Checkpoints are stored in `checkpoints/`; logs in `logs/`.

## Colab

Open `notebooks/tokenizer_bn_colab.ipynb` in Google Colab. Mount or upload your `datasets/` folder, then run cells in order.

## Configuration

Edit `configs/default.yaml` to adjust corpus sample size, vocabulary size, evaluation settings, and GPU options:

```yaml
device:
  use_gpu: true    # enable GPU metric aggregation when PyTorch + CUDA/MPS available
  device: auto     # auto, cpu, cuda, mps
  batch_size: 256
```

**Note:** SentencePiece training remains CPU-only. GPU accelerates evaluation metric aggregation and EDA statistics when PyTorch is installed.

## Project structure

```
src/tokenizer_bn/
  config.py          # YAML config loader
  device.py          # GPU/device detection and resolution
  checkpoint.py      # Resumable step tracking
  logging_utils.py   # Per-step logging
  data/              # Corpus ingestion and processing
  eda/               # Exploratory data analysis
  segmentation/      # Akshara segmentation + remap
  train/             # SentencePiece 2×2 ablation training
  tok/               # Unified tokenizer wrapper
  eval/              # Metrics, harness, plots
  run_pipeline.py    # CLI entry point
```

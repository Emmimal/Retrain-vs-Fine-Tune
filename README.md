
# Retrain vs Fine-Tune vs Train from Scratch
## Production ML Engineering — Article 08 of 15

Complete code for the article:
**"[Retrain vs Fine-Tune vs Train from Scratch: A Decision Framework for ML Engineers](https://emitechlogic.com/retrain-vs-fine-tune-vs-train-from-scratch-a-decision-framework-for-ml-engineers/)"**

Part of the [Production ML Engineering](https://emitechlogic.com/machine-learning-production-pipeline/) series at EmiTechLogic.

---

## Repository Structure

```
retrain-vs-finetune/
├── data/
│   ├── generators.py          # Synthetic dataset generators for all experiments
│   └── loaders.py             # DataLoader factory with deterministic seeding
├── models/
│   ├── base_model.py          # Shared MLP architecture used across all strategies
│   └── pretrained_registry.py # Lightweight registry — integrates with Article 04
├── strategies/
│   ├── train_from_scratch.py  # Full re-initialisation, full training loop
│   ├── fine_tune.py           # Frozen trunk + trainable head (and full unfreeze variant)
│   └── retrain.py             # Warm-start retraining on combined or new-only data
├── evaluation/
│   ├── metrics.py             # Accuracy, F1, ECE, forgetting, cost tracker
│   └── decision_framework.py  # The DecisionEngine: rule-based strategy selector
├── experiments/
│   ├── scenario_runner.py     # Runs all three experiments end-to-end
│   └── cost_model.py          # Compute + data cost estimates per strategy
├── benchmarks/
│   └── benchmark.py           # Head-to-head benchmark across all four scenarios
├── tests/
│   └── test_all.py            # 28-test suite covering all modules
├── utils/
│   └── reproducibility.py     # Seed management, snapshot utilities
├── demo.py                    # Single-file demo — run this first
└── README.md
```

---

## Quick Start

```bash
pip install torch torchvision scikit-learn numpy
python demo.py
```

## Run the Full Benchmark

```bash
python benchmarks/benchmark.py
```

## Run Tests

```bash
python -m pytest tests/test_all.py -v
```

---

## Series Navigation

- Article 03: [ML Retraining Pipeline](https://emitechlogic.com/how-to-build-an-ml-retraining-pipeline-that-wont-break-in-production/)
- Article 04: [Model Versioning](https://emitechlogic.com/model-versioning-in-production-machine-learning/)
- Article 05: [Catastrophic Forgetting in PyTorch](https://emitechlogic.com/how-to-prevent-catastrophic-forgetting-in-pytorch/)
- Article 06: [Online Learning](https://emitechlogic.com/online-learning-machine-learning-python/)
- Article 07: [Continual Learning in PyTorch](https://emitechlogic.com/continual-learning-in-pytorch/)
- **Article 08: Retrain vs Fine-Tune vs Train from Scratch ← you are here**
- Article 09: ML Model Monitoring (coming next)

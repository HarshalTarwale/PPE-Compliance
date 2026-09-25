# Real-Time PPE Compliance Detection

**Academic Deep Learning Capstone Project**

Real-time detection of Personal Protective Equipment (PPE) compliance in laboratory and workshop environments using YOLOv8.

## Detected Classes

| Class | Description |
|-------|-------------|
| Hardhat | Person wearing a hardhat |
| NO-Hardhat | Person without a hardhat |
| Safety Vest | Person wearing a safety vest |
| NO-Safety Vest | Person without a safety vest |

## Project Structure

```
ppe-compliance/
├── config/             # Centralised settings and logging configuration
├── data/               # Datasets (raw + processed) — not tracked by git
├── models/             # Trained weights and model configs
├── training/           # Training scripts and hyperparameters
├── detection/          # YOLOv8 inference and PPE detection logic
├── compliance/         # PPE compliance rule engine
├── scripts/            # CLI entry-point scripts
├── tests/              # Unit and integration tests
├── docs/               # Documentation and references
├── requirements.txt
└── README.md
```

## Setup

```bash
# 1. Clone
git clone <repo-url>
cd ppe-compliance

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux / macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify installation
python -c "from config.settings import Settings; print(Settings())"
```

## Evaluation Metrics

| Metric | Scope |
|--------|-------|
| Precision | Detection |
| Recall | Detection |
| mAP@50 | Detection |
| mAP@50-95 | Detection |
| Compliance Accuracy | Compliance |
| False-Warning Rate | Compliance |
| FPS | Performance |
| Latency (ms) | Performance |

## Roadmap

- [x] Project structure and configuration
- [ ] YOLOv8n training pipeline
- [ ] PPE detection inference
- [ ] Person–PPE association
- [ ] Compliance rule engine
- [ ] Face recognition integration
- [ ] Real-time dashboard
- [ ] Evaluation and benchmarking

## License

Academic use only.

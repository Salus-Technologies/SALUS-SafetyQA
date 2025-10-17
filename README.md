# SALUS-SafetyQA Benchmark

The first comprehensive benchmark for evaluating AI systems on construction safety knowledge.

## Overview

**1,023 questions** across 11 question types and 10 document source types.

- Multiple-choice format (4 options each)
- Covers regulations, equipment manuals, SDS, standards, and more
- Multi-jurisdictional (US states, Canadian provinces)
- Released under CC-BY-4.0

## Quick Start

```python
import json

# Load benchmark
with open('benchmark/SALUS_SafetyQA_v1.0.json', 'r') as f:
    questions = json.load(f)

# Examine a question
q = questions[0]
print(q['mc_question'])
for opt in q['mc_options']:
    marker = '✓' if opt['is_correct'] else ' '
    print(f"{marker} {opt['label']}) {opt['text']}")
```

## Dataset Statistics

- **Total Questions:** 1,023
- **Question Types:** specification (31.7%), compliance (22.5%), hazards (15.6%), procedures (9.3%), and 7 more
- **Source Types:** SDS (29.0%), standards (21.3%), regulations (20.6%), manuals (13.6%), and 6 more

## Baseline Results

| Model | Accuracy | 95% CI |
|-------|----------|--------|
| SALUS IQ | 94.04% | [92.57%, 95.41%] |
| GPT-5 | 80.25% | [77.81%, 82.70%] |
| Claude 4.1 Opus | 80.94% | [78.49%, 83.28%] |
| Claude 4.5 Sonnet | 75.07% | [72.43%, 77.71%] |
| Gemini 2.5 Pro | 78.98% | [76.44%, 81.43%] |

All improvements statistically significant (p < 0.001, McNemar's test).

## Result Availability

We provide:
- Complete benchmark dataset (1,023 questions) with ground truth
- Full evaluation harness for independent reproduction
- Aggregate statistics and bootstrap confidence intervals
- Statistical significance tests (McNemar's test)
- Visualization figures of baseline model performance

Individual model predictions and detailed error analysis are available upon request for academic research purposes under data sharing agreement. Please contact alex@salussafety.io with your research affiliation and intended use case.

## Configuration

1. Set up your API keys:
```bash
cp env.example .env
# Edit .env with your API keys
```

2. Copy the example configuration and customize as needed:
```bash
cp benchmark_config.example.yaml benchmark_config.yaml
# Edit benchmark_config.yaml with your settings
```

**Note:** Both `.env` and `benchmark_config.yaml` are gitignored to prevent accidental exposure of API keys or endpoints.

## Installation

### Using uv (recommended)

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync all dependencies (creates venv, installs deps, installs package)
uv sync

# Or with dev dependencies
uv sync --dev

# Or with all extras
uv sync --all-extras
```

### Using pip

```bash
pip install -e .
# Or with development dependencies
pip install -e ".[dev]"
# Or with all optional dependencies
pip install -e ".[all]"
```

## Evaluation

### Using installed commands (after installation)

```bash
# Evaluate a model
salus-evaluate \
  --mode direct \
  --provider openai \
  --model gpt-4o \
  --questions benchmark/SALUS_SafetyQA_v1.0.json

# Run statistical analysis
salus-analyze
```

### Using Python directly

```bash
# Evaluate a model
python evaluate_benchmark.py \
  --mode direct \
  --provider openai \
  --model gpt-4o \
  --questions benchmark/SALUS_SafetyQA_v1.0.json
```

### Evaluating Your Own RAG System

To evaluate your own retrieval-augmented generation system:

1. Implement an API endpoint following the specification in `API_SPECIFICATION.md`
2. Configure the endpoint in your `benchmark_config.yaml`
3. Run evaluation in API mode:

```bash
salus-evaluate \
  --mode api \
  --questions benchmark/SALUS_SafetyQA_v1.0.json
```

This allows fair comparison between different RAG approaches using the same evaluation protocol.

## Repository Structure

```
SALUS-SafetyQA/
├── benchmark/
│   └── SALUS_SafetyQA_v1.0.json      # 1,023 questions (1.3 MB)
├── examples/
│   └── basic_usage.py                 # Usage examples
├── figures/
│   └── fig*.png                       # Baseline result visualizations
├── evaluate_benchmark.py              # Evaluation harness
├── statistical_analysis.py            # McNemar's tests & bootstrap CIs
├── benchmark_config.example.yaml      # Example configuration
├── API_SPECIFICATION.md               # API endpoint specification for RAG evaluation
├── pyproject.toml                     # Package configuration
├── LICENSE                            # CC-BY-4.0
└── README.md                          # This file
```

**Note:** The `figures/` directory contains visualizations of baseline model performance for reference.

## Citation

```bibtex
@techreport{salus2025safetyqa,
  title={SALUS IQ: Technical White Paper \& Benchmark Report},
  author={Alex Jacobs; Dany Ayvazov},
  institution={SALUS Safety},
  year={2025},
  month={October},
  note={Version 1.0}
}
```

## Paper

Full white paper: [https://ai.salussafety.io/research](https://ai.salussafety.io/research)

## Repository

- **GitHub**: [https://github.com/Salus-Technologies/SALUS-SafetyQA](https://github.com/Salus-Technologies/SALUS-SafetyQA)

## License

CC-BY-4.0 - See LICENSE file

## Contact

- **Website:** https://salussafety.io
- **Email:** alex@salussafety.io


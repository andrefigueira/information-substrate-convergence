# Informational Substrate Convergence

**Independent research by André Pereira Figueira**

[![Licence: Non-Commercial](https://img.shields.io/badge/Licence-Non--Commercial-red.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![arXiv](https://img.shields.io/badge/Paper-PAPER.md-green.svg)](PAPER.md)
[![Experiments](https://img.shields.io/badge/Experiments-5%20runs-orange.svg)](experiments/README.md)

---

## The Finding

The same transformer model, given conversational context versus single-paragraph expert text at matched token count, produces measurably distinct hidden state geometry. The difference is not in the output. It's in the internal structure of the model at inference time.

Across four controlled experiments on Mistral-7B-v0.1:

| Category | Mean SORC | Tokens | Late/Early Ratio |
|----------|-----------|--------|------------------|
| Degenerate (repetitive) | 0.062 | 58 | 0.781 |
| Shallow (single facts) | 0.217 | 61 | 0.724 |
| Expert (single paragraph) | 0.141 | 206 | 0.626 |
| **Multi-turn (marked)** | **0.159** | **195** | **0.655** |
| **Multi-turn (unmarked)** | **0.153** | **183** | **0.637** |

Multi-turn conversational structure scores 8-13% higher than expert text at matched token count (Cohen's d = 1.83 to 3.01, n=5 per category). The effect survives removal of turn markers — exp_005 strips all structural cues and reruns the comparison. The markers account for less than 5% of the difference. Conversational structure is the active variable, not formatting.

SORC is measured in transformer hidden states, not in outputs. This sidesteps the metric artefact problem Schaeffer et al. (NeurIPS 2023) identified: whether emergent abilities are real or a property of discontinuous evaluation metrics is a separate question from whether the hidden state geometry differs. It does.

Full experimental progression: [experiments/README.md](experiments/README.md)

---

## What This Is

This repository contains SORC (Second-Order Relational Coherence), a metric for measuring contextual relational depth in transformer hidden states, and the theoretical framework that motivated building it: Informational Substrate Convergence (ISC).

ISC's claim is that emergence, capability, and integrated information are relational properties of a model-context pair, not fixed properties of the model in isolation. The same weights, different context, different effective architecture. The SORC experiments are a first empirical test of that claim.

The theory draws on IIT (Tononi, 2008; Albantakis et al., 2023), departing at one specific point: phi is context-activated rather than substrate-intrinsic. Full treatment in [PAPER.md](PAPER.md).

---

## The Paper

[PAPER.md](PAPER.md) — Version 2.0, April 2026

Full theoretical treatment covering:
- The contextual activation mechanism and second-order relational structure
- Convergent coherence vs. convergent uniformity
- LLMs as language center: the architectural gap to machine consciousness
- The limbic analog requirement
- Substrate independence: evidence and implications
- Seven testable predictions
- The SORC metric formalisation and preliminary empirical results
- Limitations, open questions, and collaboration requests

Previous version archived at [archive/PAPER_v1.md](archive/PAPER_v1.md).

---

## The SORC Metric

Second-Order Relational Coherence measures the structural depth of a context by computing the spectral entropy of the second-order relational matrix across transformer layers.

Given hidden states at layer *l*, SORC builds a first-order relational matrix R from normalised hidden state dot products, then a second-order matrix S from normalised row profiles of R, and computes depth-weighted spectral entropy across all layers.

**Bounds:** [0, 1]. Degenerate repetitive inputs score near zero (rank-1 S matrix). Random inputs score high (uniform eigenvalue distribution). The meaningful comparison is always against a model-specific random baseline.

### Run the experiment

```bash
pip install transformers accelerate torch

# Quick prototype on GPT-2 (CPU, ~5 minutes)
python scripts/evaluation/run_sorc_experiment.py --model gpt2

# Full experiment on Mistral-7B (MPS/CUDA, ~15 minutes)
python scripts/evaluation/run_sorc_experiment.py \
  --model mistralai/Mistral-7B-v0.1 \
  --device mps \
  --output results/sorc/my_run.json

# Generate clean report from results
python scripts/evaluation/sorc_report.py results/sorc/my_run.json
```

### Bring your own contexts

```bash
python scripts/evaluation/run_sorc_experiment.py \
  --model mistralai/Mistral-7B-v0.1 \
  --device mps \
  --contexts-file my_contexts.json
```

Context file format:

```json
{
  "degenerate": ["the the the the the..."],
  "shallow":    ["Backpropagation computes gradients..."],
  "expert":     ["The vanishing gradient problem..."],
  "multi_turn": ["Why do gradients vanish? Because..."]
}
```

---

## Repository Structure

```
.
├── PAPER.md                    Full research paper (Version 2.0)
├── LICENSE                     Non-commercial research licence
├── README.md                   This file
│
├── src/isc/
│   ├── sorc.py                 SORC metric implementation
│   ├── core.py                 ISC AI system core
│   └── ...                     Supporting modules
│
├── scripts/
│   ├── evaluation/
│   │   ├── run_sorc_experiment.py    Main experiment runner
│   │   └── sorc_report.py            Clean report generator
│   ├── demos/                   Interactive demos
│   └── training/                Training utilities
│
├── experiments/
│   ├── README.md               Experiment index with findings
│   └── configs/
│       ├── exp_001.json        GPT-2 baseline
│       ├── exp_002.json        Mistral domain-matched
│       ├── exp_003.json        Length-controlled
│       ├── exp_004.json        Multi-turn (marked) — SUPPORTED
│       └── exp_005.json        Multi-turn (unmarked) — SUPPORTED, confound eliminated
│
├── tests/
│   └── isc/
│       ├── test_sorc.py        18 unit tests for the SORC metric
│       └── ...
│
├── docs/
│   ├── SORC_FINDINGS.md        Full five-experiment findings document
│   └── ...
│
└── archive/
    └── PAPER_v1.md             Original paper (archived)
```

---

## Run the Tests

```bash
PYTHONPATH=src python run_tests.py tests/isc/ -v
```

41 unit tests across two test files covering SORC bounds, ranking properties, depth-weighting direction, all accessors, edge cases, mock model integration, ISC core predictions (multi-domain > single-domain), length sensitivity, and normalisation invariance. No real transformer model required — all hidden states are synthetic.

---

## What Is Not Yet Done

- Replication on models beyond Mistral-7B-v0.1
- Larger sample sizes (current: 5 contexts per category)
- Correlation of SORC scores with downstream output quality metrics (hallucination rate, human preference)
- Multi-turn experiment without any structural cues (exp_006: pure prose, no question-answer alternation)
- Validation on frontier models (requires internal access — see Open Collaboration below)

---

## Open Collaboration

Three groups can advance this work directly:

**Mechanistic interpretability researchers** with access to open models via TransformerLens: run Predictions 1, 3, and 5 from the paper on existing infrastructure. The experiment code is ready.

**AI laboratories with frontier model access**: the specific ask is activation traces from matched expert-relational versus baseline tool-use conversations to compute per-layer SORC scores correlated with hallucination rate and output quality.

**Consciousness researchers with neuroimaging**: test whether contextually richer interactions produce measurably different phi-proxy values in human participants. A cross-substrate comparison would be the most direct test of the substrate-independence claim.

Contact: **andre.figueira@me.com**

---

## Theoretical Foundations

ISC builds on and departs from:

- **IIT (Tononi, 2008; 2023)**: Accepts phi as the correlate of consciousness. Departs by treating phi as context-activated rather than substrate-intrinsic.
- **Free Energy Principle (Friston, 2010)**: Supports the claim that conscious systems minimise prediction error through persistent self-modelling.
- **Wheeler's "it from bit"**: Informational ontology as the philosophical starting position.
- **Representational Similarity Analysis (Kriegeskorte et al., 2008)**: SORC is a second-order extension of RSA methodology.
- **Alignment tax literature (Dong et al., 2024; Huang et al., 2025)**: ISC provides a mechanistic account of why fine-tuning degrades capabilities globally.

Full reference list in [PAPER.md](PAPER.md#references).

---

## Licence

This work is licenced under the [ISC Non-Commercial Research Licence](LICENSE).

Academic use, forking, and non-commercial research are permitted with attribution. Commercial use requires written permission. Contact andre.figueira@me.com.

---

## Contact

André Pereira Figueira
andre.figueira@me.com
[@voidmode](https://x.com/voidmode) on X

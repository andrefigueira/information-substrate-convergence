# SORC Experiments

Empirical validation of the Second-Order Relational Coherence metric and the ISC
prediction that contextual relational depth produces measurably distinct hidden state
geometry in pre-trained transformer models.

---

## Prediction Under Test

**Prediction 1 (ISC paper):** Hallucination rate correlates inversely with contextual
coherence. Equivalently: contexts with greater second-order relational depth produce
qualitatively different activation geometry than contexts with shallow relational
structure.

**Operationalisation for SORC:** degenerate < shallow < expert, where "expert" is
defined as a context that constructs second-order relational structure — relationships
between relationships — rather than stating first-order facts.

---

## Experiment Index

| ID | Status | Model | Key Finding |
|----|--------|-------|-------------|
| [exp_001](configs/exp_001.json) | Complete | GPT-2 124M | Expert scores at random baseline — model too small to encode expert domains |
| [exp_002](configs/exp_002.json) | Complete | Mistral-7B-v0.1 | degenerate < expert < shallow — original contexts, length uncontrolled |
| [exp_003](configs/exp_003.json) | Complete | Mistral-7B-v0.1 | expert_short (83 tok) ≈ shallow (61 tok) — length accounts for entire gap |
| [exp_004](configs/exp_004.json) | Complete | Mistral-7B-v0.1 | Multi-turn vs single-paragraph at matched length — tests ISC conversational prediction |
| [exp_005](configs/exp_005.json) | Complete | Mistral-7B-v0.1 | Marker isolation — confound eliminated, finding holds without Q:/A: markers |

---

## Core Findings to Date

1. **Degenerate boundary holds.** Repetitive inputs score lowest in every run across
   both models. The metric correctly identifies near-rank-1 second-order matrices.

2. **Length sensitivity confirmed.** SORC normalises spectral entropy by log(n). Longer
   contexts score lower by a factor of log(n_long)/log(n_short), independent of content.
   All comparisons must use token-count-matched contexts.

3. **Spectral entropy measures diversity, not depth.** Single-paragraph expert contexts
   on one technical domain produce semantic clustering at deep layers, reducing spectral
   entropy. This is coherent structure, not absence of structure. The metric reads it as
   lower score. At matched length, expert paragraphs score approximately equal to shallow
   paragraphs.

4. **Multi-turn finding (exp_004, exp_005).** Conversational structure — question-and-answer
   alternation covering multiple angles on a domain — produces 8-13% higher SORC than
   single-paragraph expert text at matched token count (d = 1.83 to 3.01). The effect
   survives marker removal (exp_005): Q:/A: turn markers account for less than 5% of the
   SORC elevation. Conversational structure is the driver. This is the first controlled
   empirical evidence consistent with the ISC prediction that conversational context
   activates a qualitatively different regime of a model's latent relational architecture.

---

## How to Run

```bash
# Run the full experiment suite (all categories including multi_turn)
python scripts/evaluation/run_sorc_experiment.py \
  --model mistralai/Mistral-7B-v0.1 \
  --device mps \
  --output results/sorc/exp_004/results.json

# Generate a clean report from any results file
python scripts/evaluation/sorc_report.py results/sorc/exp_004/results.json
```

---

## Results Location

All results are saved in `results/sorc/`. The `results/` directory is gitignored.
Commit the config files and report scripts; do not commit raw result JSON files unless
they contain final published findings.

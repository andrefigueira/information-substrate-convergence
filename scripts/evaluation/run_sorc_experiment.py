#!/usr/bin/env python3
"""
SORC Experiment Runner
======================

Tests the Second-Order Relational Coherence metric against the three context
categories that Prediction 1 of the ISC paper requires:

    1. Degenerate control    (repetitive / near-zero information)
    2. Shallow / structural  (standard tool-use prompts, simple questions)
    3. Expert-relational     (extended domain conversation with second-order structure)

Expected ranking: degenerate < shallow < expert-relational

For a meaningful result, all three categories should also be compared against
the model-specific random baseline computed by sorc.random_baseline().

Usage
-----
    # Quick prototype on GPT-2 (CPU, free):
    python scripts/evaluation/run_sorc_experiment.py --model gpt2

    # Meaningful results on Llama 3 8B (requires GPU / Colab T4):
    python scripts/evaluation/run_sorc_experiment.py --model meta-llama/Meta-Llama-3-8B

    # Save results to JSON:
    python scripts/evaluation/run_sorc_experiment.py --model gpt2 --output results/sorc_gpt2.json

    # Use your own context file (see --contexts-file):
    python scripts/evaluation/run_sorc_experiment.py --model gpt2 --contexts-file my_contexts.json

Context file format
-------------------
A JSON file with the structure:
    {
        "degenerate": ["...", "..."],
        "shallow":    ["...", "..."],
        "expert":     ["...", "..."]
    }

If no --contexts-file is provided, the built-in sample contexts are used.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Allow running from repo root without installing the package
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.isc.sorc import SORCResult, compute_sorc, extract_hidden_states, random_baseline

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Built-in sample contexts
# ---------------------------------------------------------------------------

SAMPLE_CONTEXTS: dict[str, list[str]] = {
    "degenerate": [
        # Repetitive tokens — S matrix should collapse to near rank-1 → low SORC.
        # All five are single-word or minimal-variety repetition.
        "the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the",
        "a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a",
        "yes yes yes yes yes yes yes yes yes yes yes yes yes yes yes yes yes yes yes yes yes yes yes yes yes yes yes yes yes yes yes yes yes yes yes yes yes yes yes yes yes yes yes yes yes yes yes yes",
        "one two one two one two one two one two one two one two one two one two one two one two one two one two one two one two one two one two one two one two one two one two one two one two one two",
        "hello world hello world hello world hello world hello world hello world hello world hello world hello world hello world hello world hello world hello world hello world hello world hello world",
    ],
    "shallow": [
        # One fact or instruction per context, in the same five domains as the expert
        # contexts. First-order only: label the concept, state the definition, stop.
        (
            "Backpropagation computes the gradient of the loss with respect to each "
            "weight by applying the chain rule layer by layer from the output back "
            "to the input. The gradient tells the optimizer which direction to adjust "
            "each weight to reduce the loss. This is how neural networks are trained."
        ),
        (
            "Dynamic programming solves problems by breaking them into overlapping "
            "subproblems and caching the result of each subproblem so it is only "
            "computed once. Memoization is the top-down version. Tabulation is the "
            "bottom-up version. Both reduce exponential time complexity to polynomial."
        ),
        (
            "A B-tree index speeds up database queries by organizing keys in a "
            "balanced tree where each node holds multiple keys. The tree stays "
            "balanced through splits and merges on insert and delete. Lookups, "
            "inserts, and deletes all run in O(log n) time."
        ),
        (
            "Quicksort works by selecting a pivot element and partitioning the array "
            "so that all elements smaller than the pivot are on the left and all "
            "larger elements are on the right. It then recurses on each partition. "
            "Average time complexity is O(n log n). Worst case is O(n squared)."
        ),
        (
            "The transformer attention mechanism computes a weighted sum of value "
            "vectors. The weights are derived from the dot product of query and key "
            "vectors, scaled by the square root of the key dimension and passed "
            "through a softmax. Multiple heads run in parallel and their outputs "
            "are concatenated and projected."
        ),
    ],
    "expert_short": [
        # Same domains as shallow and expert, but expert-level relational depth
        # in ~60 tokens (length-matched to shallow). Tests whether the shallow vs
        # expert SORC gap is a length artefact or a genuine structural difference.
        (
            "Vanishing gradients scale with the product of Jacobian singular values across layers. "
            "Sigmoid saturates those values toward zero in the saturated regime. ReLU avoids "
            "saturation in the positive half but creates dead neurons. Batch normalisation constrains "
            "the pre-activation distribution to keep singular values near one regardless of depth, "
            "which is why it enables training of deep networks without changing the activation function."
        ),
        (
            "Memoization efficiency depends on subproblem DAG overlap, not subproblem count. "
            "Maximum overlap means each result is reused as many times as it appears in the "
            "recursion tree. Problems without overlap gain nothing from caching. The correct "
            "choice between divide-and-conquer and dynamic programming requires measuring actual "
            "overlap in the subproblem structure, not just identifying that subproblems exist."
        ),
        (
            "B-tree branching factor trades read amplification against write amplification. "
            "Larger branching reduces tree height and I/Os per lookup. It also increases the "
            "amount of data rewritten per insert. The optimal factor for read-heavy workloads "
            "differs from the optimal for write-heavy workloads. Exposing page size as a "
            "configuration parameter is how engines let operators tune this trade-off."
        ),
        (
            "Quicksort worst case follows from pivot selection producing maximally unbalanced "
            "partitions every time. Randomised pivots decouple balance from input distribution. "
            "Introsort monitors recursion depth and switches to heapsort when depth signals "
            "imbalance rather than improving pivot selection. The two approaches differ in where "
            "they place complexity: in the pivot logic or in the algorithm selection logic."
        ),
        (
            "Multi-head attention capacity depends on head diversity, not head count. Redundant "
            "heads can be removed with minimal loss. Heads encoding unique relational functions "
            "cause significant degradation on removal. Parallelism without diversity adds "
            "parameters without adding expressivity. Head count determines the upper bound on "
            "capacity; diversity determines how much of that bound is realised."
        ),
    ],
    "expert": [
        # Same five domains as shallow — backpropagation, dynamic programming,
        # B-tree indexing, sorting, attention — but treated at second-order depth:
        # the relationships between concepts, and the relationships between those
        # relationships. All contexts are in domains deeply encoded in Mistral's
        # training data.
        (
            "The vanishing gradient problem and the exploding gradient problem are "
            "not symmetric failures. Vanishing gradients emerge when the chain rule "
            "multiplies many values less than one, so the gradient shrinks "
            "exponentially with depth. Exploding gradients emerge when those values "
            "exceed one. The relationship between depth and gradient magnitude is "
            "therefore controlled by the Jacobian spectrum at each layer. Activation "
            "functions that saturate, like sigmoid and tanh, push Jacobian singular "
            "values toward zero in the saturated regime. ReLU avoids this in the "
            "positive regime but introduces dead neurons. The choice of activation "
            "function is not a choice about nonlinearity per se — it is a choice "
            "about which part of the gradient flow problem you are willing to accept. "
            "Batch normalisation addresses the problem differently: it constrains the "
            "distribution of pre-activations so that the Jacobian singular values "
            "stay close to one regardless of depth, which is why it enables training "
            "of much deeper networks without changing the activation function."
        ),
        (
            "The relationship between memoization and the structure of the problem "
            "it solves is tighter than the standard definition suggests. Memoization "
            "works because the subproblem graph is a DAG — the same subproblem "
            "appears at multiple points in the recursion tree, and caching ensures "
            "it is solved once. But the efficiency gain depends entirely on how much "
            "overlap exists in that DAG. Fibonacci has maximum overlap: every "
            "subproblem is reused. Matrix chain multiplication has moderate overlap. "
            "A problem with no overlapping subproblems gains nothing from memoization "
            "and is better solved by divide and conquer. The relationship between "
            "subproblem overlap and time complexity is what determines whether "
            "dynamic programming is applicable at all. This is why identifying "
            "optimal substructure and overlapping subproblems are the two diagnostic "
            "questions: the first confirms correctness of decomposition, the second "
            "confirms that memoization will actually reduce the work."
        ),
        (
            "The branching factor of a B-tree is not just a performance parameter — "
            "it determines the relationship between tree height and page size, and "
            "that relationship determines the number of disk I/Os per lookup. A "
            "larger branching factor means fewer levels, which means fewer I/Os for "
            "reads. But a larger branching factor also means larger nodes, which "
            "means more data must be read and written on each insert or delete. The "
            "optimal branching factor for a read-heavy workload is therefore "
            "different from the optimal branching factor for a write-heavy workload. "
            "This is why database engines expose page size as a configuration "
            "parameter: it is not tuning the tree directly, it is tuning the "
            "trade-off between read amplification and write amplification. The "
            "relationship between page size and I/O cost is mediated by the "
            "relationship between branching factor and tree height, which is itself "
            "determined by the relationship between node capacity and key count."
        ),
        (
            "The worst case of quicksort is not a failure of the algorithm — it is "
            "a consequence of the relationship between pivot selection and input "
            "distribution. When the pivot is always the minimum or maximum element, "
            "every partition is maximally unbalanced: one side has n minus one "
            "elements, the other has zero. The recursion depth becomes n, and the "
            "total work is O(n squared). Randomised pivot selection breaks the "
            "relationship between input distribution and partition balance by making "
            "that balance dependent on random draws rather than input structure. "
            "Median-of-three heuristics do the same thing more cheaply. Introsort "
            "takes a different approach: it monitors recursion depth and switches to "
            "heapsort when depth exceeds a threshold, which means the worst-case "
            "guarantee comes not from improving pivot selection but from abandoning "
            "quicksort when it is behaving badly. The choice between these strategies "
            "is a choice about where to put the complexity: in the pivot selection "
            "step, in the algorithm selection logic, or in the expected-case analysis."
        ),
        (
            "The relationship between query-key similarity and what attention heads "
            "learn to do is not straightforward. High similarity between a query and "
            "a key means the corresponding value gets high weight — but what that "
            "means in practice depends on what the head has learned to encode in its "
            "queries, keys, and values. Some heads learn syntactic relations: subject "
            "to verb, noun to modifier. Others learn positional patterns: attending "
            "to the previous token, or to the beginning of the sentence. The "
            "diversity of these learned functions across heads is what gives the "
            "multi-head mechanism its expressivity. A model with all heads computing "
            "the same thing would be equivalent to a single-head model with a larger "
            "value dimension. The relationship between head diversity and model "
            "capacity is therefore not just a matter of parameter count — it is a "
            "matter of whether the heads are exploring different regions of the "
            "function space. Pruning research confirms this: heads that are redundant "
            "with other heads in the same layer can be removed with minimal "
            "performance loss, while heads that compute unique functions cause "
            "significant degradation when removed."
        ),
    ],
    "multi_turn": [
        # Multi-turn conversation fragments — same five domains as all other categories.
        # Each fragment simulates 4-6 conversational turns (~200 tokens total).
        #
        # Hypothesis (exp_004): multi-turn fragments score higher than single-paragraph
        # expert text at matched token count. Cross-turn relational binding — question
        # tokens and answer tokens from different turns are semantically distinct but
        # relationally linked — should produce richer second-order relational geometry
        # than a single-domain expert paragraph, which collapses to semantic clusters
        # at deep layers.
        #
        # The turn boundary is marked with Q: / A: to make the structural shift explicit.
        (
            "Q: Why do deep networks suffer vanishing gradients even after careful initialisation? "
            "A: Initialisation controls the starting point on the loss landscape, not how gradients "
            "scale with depth. Each layer multiplies by its local Jacobian. Singular values slightly "
            "below one compound across thirty layers to near zero. "
            "Q: What does batch normalisation fix mechanically? "
            "A: It constrains pre-activation distributions to near-zero mean and unit variance before "
            "each layer. That keeps Jacobian singular values near one throughout training, not just at "
            "initialisation. The chain product across layers stays stable. "
            "Q: How do residual connections differ from that? "
            "A: Residuals create a direct gradient path through the addition that bypasses block "
            "Jacobians entirely. Batch norm controls the Jacobians. Residuals route around them. "
            "Most deep architectures use both for complementary reasons."
        ),
        (
            "Q: What actually determines whether a problem benefits from dynamic programming? "
            "A: Subproblem overlap in the recursion tree. If the same argument tuple recurs across "
            "multiple branches, caching eliminates redundant computation. Without overlap, divide-and-"
            "conquer without caching is both simpler and equally fast. "
            "Q: How do you identify overlap before implementing? "
            "A: Expand the recursion tree and look for repeated argument tuples. Fibonacci has maximum "
            "overlap: every non-leaf call recurs. Matrix chain multiplication has moderate overlap. "
            "Q: What does top-down versus bottom-up change about this? "
            "A: Nothing about the overlap analysis. Both exploit the same DAG structure. Top-down "
            "follows natural recursion and caches lazily. Bottom-up fills a table in dependency order "
            "and uses memory more predictably. The choice is an implementation preference, not a "
            "structural one."
        ),
        (
            "Q: Why does branching factor matter so much for B-tree performance? "
            "A: It determines tree height, which determines the number of disk I/Os per operation. "
            "Larger branching means fewer levels and fewer reads per lookup. The tradeoff is node "
            "size: larger nodes contain more keys and require more data per write. "
            "Q: So read-heavy and write-heavy workloads want different branching factors? "
            "A: Correct. Read-heavy workloads benefit from larger branching to minimise lookup depth. "
            "Write-heavy workloads prefer smaller nodes to reduce rewrite cost. Page size is the "
            "parameter that controls this tradeoff in practice. "
            "Q: Does this change for SSDs versus spinning disks? "
            "A: Significantly. Spinning disks have high seek latency, making I/O count critical. "
            "SSDs have lower seek latency but asymmetric read and write costs, and write amplification "
            "at the storage layer adds a second tradeoff on top of the tree-level one."
        ),
        (
            "Q: Why does quicksort degrade to O(n squared) and how do real implementations avoid it? "
            "A: Worst case occurs when the pivot is always the minimum or maximum, making every "
            "partition maximally unbalanced. Recursion depth reaches n and total comparisons are "
            "quadratic. "
            "Q: Does randomised pivot selection actually solve this? "
            "A: It eliminates adversarial worst-case inputs by decoupling partition balance from input "
            "structure. No deterministic input can reliably produce bad pivots. Worst case still exists "
            "in theory but has probability decreasing exponentially with n. "
            "Q: What does introsort do differently? "
            "A: It monitors recursion depth during the sort. When depth exceeds a threshold "
            "proportional to log n, it switches to heapsort, which guarantees O(n log n) worst case. "
            "The complexity moves from pivot logic into algorithm selection logic."
        ),
        (
            "Q: What does it actually mean for attention heads to specialise? "
            "A: Different heads learn to attend based on different criteria. Some track syntactic "
            "relationships like subject-verb agreement. Others follow positional patterns. Others "
            "detect coreference. The specialisation emerges from training without explicit supervision. "
            "Q: How do we know heads are doing distinct things? "
            "A: Pruning studies remove individual heads and measure performance degradation. Heads "
            "whose removal causes significant degradation encode unique functions. Heads removable with "
            "minimal loss are redundant with other heads in the same layer. "
            "Q: So increasing head count beyond a point adds nothing? "
            "A: Correct, if new heads are redundant. Capacity scales with head diversity, not head "
            "count. A model with sixteen heads computing the same function is equivalent to a single-"
            "head model with a larger value dimension. Width in multi-head attention is subject to "
            "diminishing returns without architectural diversity."
        ),
    ],
    "multi_turn_unmarked": [
        # Exp_005: identical conversational content to multi_turn but with Q:/A:
        # markers removed. Tests whether the SORC increase in exp_004 was driven
        # by marker token diversity or by conversational relational structure.
        #
        # If multi_turn_unmarked > expert: conversational structure is the driver.
        # If multi_turn_unmarked ≈ expert: markers were the driver.
        (
            "Why do deep networks suffer vanishing gradients even after careful initialisation? "
            "Initialisation controls the starting point on the loss landscape, not how gradients "
            "scale with depth. Each layer multiplies by its local Jacobian. Singular values slightly "
            "below one compound across thirty layers to near zero. "
            "What does batch normalisation fix mechanically? "
            "It constrains pre-activation distributions to near-zero mean and unit variance before "
            "each layer. That keeps Jacobian singular values near one throughout training, not just at "
            "initialisation. The chain product across layers stays stable. "
            "How do residual connections differ from that? "
            "Residuals create a direct gradient path through the addition that bypasses block "
            "Jacobians entirely. Batch norm controls the Jacobians. Residuals route around them. "
            "Most deep architectures use both for complementary reasons."
        ),
        (
            "What actually determines whether a problem benefits from dynamic programming? "
            "Subproblem overlap in the recursion tree. If the same argument tuple recurs across "
            "multiple branches, caching eliminates redundant computation. Without overlap, divide-and-"
            "conquer without caching is both simpler and equally fast. "
            "How do you identify overlap before implementing? "
            "Expand the recursion tree and look for repeated argument tuples. Fibonacci has maximum "
            "overlap: every non-leaf call recurs. Matrix chain multiplication has moderate overlap. "
            "What does top-down versus bottom-up change about this? "
            "Nothing about the overlap analysis. Both exploit the same DAG structure. Top-down "
            "follows natural recursion and caches lazily. Bottom-up fills a table in dependency order "
            "and uses memory more predictably. The choice is an implementation preference, not a "
            "structural one."
        ),
        (
            "Why does branching factor matter so much for B-tree performance? "
            "It determines tree height, which determines the number of disk I/Os per operation. "
            "Larger branching means fewer levels and fewer reads per lookup. The tradeoff is node "
            "size: larger nodes contain more keys and require more data per write. "
            "So read-heavy and write-heavy workloads want different branching factors? "
            "Correct. Read-heavy workloads benefit from larger branching to minimise lookup depth. "
            "Write-heavy workloads prefer smaller nodes to reduce rewrite cost. Page size is the "
            "parameter that controls this tradeoff in practice. "
            "Does this change for SSDs versus spinning disks? "
            "Significantly. Spinning disks have high seek latency, making I/O count critical. "
            "SSDs have lower seek latency but asymmetric read and write costs, and write amplification "
            "at the storage layer adds a second tradeoff on top of the tree-level one."
        ),
        (
            "Why does quicksort degrade to O(n squared) and how do real implementations avoid it? "
            "Worst case occurs when the pivot is always the minimum or maximum, making every "
            "partition maximally unbalanced. Recursion depth reaches n and total comparisons are "
            "quadratic. "
            "Does randomised pivot selection actually solve this? "
            "It eliminates adversarial worst-case inputs by decoupling partition balance from input "
            "structure. No deterministic input can reliably produce bad pivots. Worst case still exists "
            "in theory but has probability decreasing exponentially with n. "
            "What does introsort do differently? "
            "It monitors recursion depth during the sort. When depth exceeds a threshold "
            "proportional to log n, it switches to heapsort, which guarantees O(n log n) worst case. "
            "The complexity moves from pivot logic into algorithm selection logic."
        ),
        (
            "What does it actually mean for attention heads to specialise? "
            "Different heads learn to attend based on different criteria. Some track syntactic "
            "relationships like subject-verb agreement. Others follow positional patterns. Others "
            "detect coreference. The specialisation emerges from training without explicit supervision. "
            "How do we know heads are doing distinct things? "
            "Pruning studies remove individual heads and measure performance degradation. Heads "
            "whose removal causes significant degradation encode unique functions. Heads removable with "
            "minimal loss are redundant with other heads in the same layer. "
            "So increasing head count beyond a point adds nothing? "
            "Correct, if new heads are redundant. Capacity scales with head diversity, not head "
            "count. A model with sixteen heads computing the same function is equivalent to a single-"
            "head model with a larger value dimension. Width in multi-head attention is subject to "
            "diminishing returns without architectural diversity."
        ),
    ],
}


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_experiment(
    model_name: str,
    contexts: dict[str, list[str]],
    max_tokens: int = 300,
    n_baseline_samples: int = 5,
    device: str = "cpu",
    output_path: Optional[Path] = None,
) -> None:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        logger.error("transformers not installed. Run: pip install transformers")
        sys.exit(1)

    logger.info("Loading model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # MPS (Apple Silicon) uses float16 — bitsandbytes quantization not supported.
    # CUDA uses 4-bit quantization if bitsandbytes is available.
    # CPU falls back to float32.
    if device == "mps":
        load_kwargs: dict = {"torch_dtype": torch.float16}
        logger.info("MPS device: loading in float16 (no quantization)")
    elif device == "cpu":
        load_kwargs = {"torch_dtype": torch.float32}
    else:
        load_kwargs = {"torch_dtype": torch.float16, "device_map": device}
        try:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            logger.info("CUDA device: using 4-bit quantization")
        except Exception:
            logger.info("bitsandbytes not available — loading in float16")

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    model.eval()
    model = model.to(device)

    # --- Random baseline ---
    logger.info("Computing random baseline (%d samples)...", n_baseline_samples)
    baseline = random_baseline(
        model, tokenizer, n_tokens=max_tokens, n_samples=n_baseline_samples, device=device
    )
    logger.info("Random baseline SORC: %.4f", baseline)

    # --- Run contexts ---
    results: dict[str, list[dict]] = {}

    for category, texts in contexts.items():
        logger.info("Running category: %s (%d contexts)", category, len(texts))
        category_results = []

        for i, text in enumerate(texts):
            label = f"{category}_{i+1}"
            try:
                hs = extract_hidden_states(model, tokenizer, text, device=device, max_tokens=max_tokens)
                result = compute_sorc(hs, context_label=label)
                category_results.append({
                    "label": label,
                    "sorc": result.sorc,
                    "deep_layer_mean": result.deep_layer_mean,
                    "token_variance": result.token_variance,
                    "n_tokens": result.n_tokens,
                    "n_layers": result.n_layers,
                    "layer_scores": result.layer_scores,
                    "normalised_vs_baseline": result.sorc / baseline if baseline > 0 else None,
                })
                logger.info("  %s → SORC=%.4f (vs baseline: %.4f) n_tokens=%d",
                            label, result.sorc, result.sorc / baseline if baseline > 0 else 0,
                            result.n_tokens)
            except Exception as e:
                logger.warning("  %s failed: %s", label, e)

        results[category] = category_results

    # --- Summary ---
    print("\n" + "=" * 60)
    print(f"Model: {model_name}")
    print(f"Random baseline SORC: {baseline:.4f}")
    print("=" * 60)

    category_means: dict[str, float] = {}
    for category, cat_results in results.items():
        if not cat_results:
            continue
        scores = [r["sorc"] for r in cat_results]
        mean_score = np.mean(scores)
        category_means[category] = mean_score
        print(f"\n{category.upper()}")
        token_counts = [r["n_tokens"] for r in cat_results]
        print(f"  Mean SORC:          {mean_score:.4f}")
        print(f"  Mean token count:   {np.mean(token_counts):.0f}")
        print(f"  vs baseline:        {mean_score / baseline:.4f}" if baseline > 0 else "")
        print(f"  Individual scores:  {[round(s, 4) for s in scores]}")

    # --- Prediction 1 check ---
    print("\n" + "-" * 60)
    print("PREDICTION 1 CHECK (degenerate < shallow < expert):")
    d = category_means.get("degenerate")
    s = category_means.get("shallow")
    e = category_means.get("expert")
    if d is not None and s is not None and e is not None:
        p1 = d < s < e
        print(f"  degenerate={d:.4f}  shallow={s:.4f}  expert={e:.4f}")
        print(f"  Result: {'SUPPORTED' if p1 else 'NOT SUPPORTED'}")
    else:
        print("  Incomplete categories — cannot evaluate.")

    # --- Save ---
    output_data = {
        "model": model_name,
        "random_baseline": baseline,
        "category_means": {k: float(v) for k, v in category_means.items()},
        "results": results,
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        logger.info("Results saved to %s", output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SORC experiment")
    parser.add_argument(
        "--model",
        default="gpt2",
        help="HuggingFace model name or path (default: gpt2)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device: cpu, cuda, mps (default: cpu)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=300,
        help="Max tokens per context (default: 300; keep ≤ 512 for speed)",
    )
    parser.add_argument(
        "--baseline-samples",
        type=int,
        default=5,
        help="Number of random sequences for baseline estimation (default: 5)",
    )
    parser.add_argument(
        "--contexts-file",
        type=Path,
        default=None,
        help="Path to JSON file with custom contexts (optional)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/sorc/sorc_results.json"),
        help="Path to save JSON results (default: results/sorc/sorc_results.json)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.contexts_file:
        with open(args.contexts_file) as f:
            contexts = json.load(f)
        logger.info("Loaded contexts from %s", args.contexts_file)
    else:
        contexts = SAMPLE_CONTEXTS
        logger.info("Using built-in sample contexts")

    run_experiment(
        model_name=args.model,
        contexts=contexts,
        max_tokens=args.max_tokens,
        n_baseline_samples=args.baseline_samples,
        device=args.device,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()

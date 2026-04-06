"""
exp_007: SORC across SUSAN experimental conditions.

Runs four conditions (control, history_only, feedback_blind, self_referential)
on five multi-turn reasoning scenarios using Mistral-7B. Measures SORC on the
full input context at each turn and saves all results to JSON.

Usage:
    python -m experiments.susan_sorc.run [--model MODEL] [--device DEVICE] [--output PATH]

Defaults:
    model:  mistralai/Mistral-7B-v0.1
    device: mps  (auto-detects cuda > mps > cpu)
    output: results/exp_007_results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

# Allow running as a script or module from repo root
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from isc.sorc import SORCResult, compute_sorc, extract_hidden_states, random_baseline  # noqa: E402

from experiments.susan_sorc.conditions import CONDITIONS, ContextResult, build_context  # noqa: E402
from experiments.susan_sorc.monitor import Monitor, MonitorScores  # noqa: E402
from experiments.susan_sorc.regulator import Regulator  # noqa: E402
from experiments.susan_sorc.scenarios import SCENARIOS, Scenario  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("exp_007")

MAX_CONTEXT_TOKENS = 512
MAX_NEW_TOKENS = 200
DEFAULT_TEMPERATURE = 0.7


@dataclass
class TurnResult:
    condition: str
    scenario_id: str
    turn: int  # 0-indexed
    sorc: float
    deep_layer_mean: float
    token_variance: float
    n_tokens: int  # context token count (length covariate)
    n_layers: int
    coherence: float
    goal_alignment: float
    reasoning_depth: float
    temperature: float
    context_retention: float
    n_history_turns: int
    context_length_chars: int
    response_length_words: int
    elapsed_seconds: float
    response_text: str
    prompt_text: str


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_name: str, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info("Loading tokenizer: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log.info("Loading model on %s (this may take a minute)...", device)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device in ("cuda", "mps") else torch.float32,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    log.info(
        "Model loaded. Parameters: %dM", sum(p.numel() for p in model.parameters()) // 1_000_000
    )
    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    context: str,
    temperature: float,
    device: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    """Generate the model's response to a context string."""
    inputs = tokenizer(
        context,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_CONTEXT_TOKENS,
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0.01,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    new_ids = output_ids[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def run_scenario_condition(
    scenario: Scenario,
    condition: str,
    model,
    tokenizer,
    monitor: Monitor,
    device: str,
) -> list[TurnResult]:
    """Run one scenario under one condition, returning a TurnResult per turn."""
    regulator = Regulator()
    results: list[TurnResult] = []
    prev_prompts: list[str] = []
    prev_responses: list[str] = []
    prev_scores: MonitorScores | None = None
    coherence_trend: list[float] = []

    log.info("  condition=%-18s scenario=%s", condition, scenario.id)

    for turn_idx, prompt in enumerate(scenario.turns):
        t0 = time.perf_counter()

        # --- Build context for this condition ---
        ctx: ContextResult = build_context(
            condition=condition,
            turn_prompts=prev_prompts,
            turn_responses=prev_responses,
            current_prompt=prompt,
            scores=prev_scores,
            monitor_note=(
                monitor.generate_note(
                    prev_scores, coherence_trend[-2] if len(coherence_trend) >= 2 else None
                )
                if prev_scores is not None
                else "No prior turn."
            ),
            coherence_trend=coherence_trend[-3:],
            regulator_status=regulator.status_line(),
            context_retention=regulator.context_retention,
        )

        # --- Measure SORC on the context (before generation) ---
        hidden_states = extract_hidden_states(
            model, tokenizer, ctx.text, device=device, max_tokens=MAX_CONTEXT_TOKENS
        )
        sorc_result: SORCResult = compute_sorc(
            hidden_states,
            context_label=f"{condition}_{scenario.id}_t{turn_idx}",
        )

        # --- Generate response ---
        temperature = (
            regulator.temperature
            if condition in ("feedback_blind", "self_referential")
            else DEFAULT_TEMPERATURE
        )
        response = generate_response(model, tokenizer, ctx.text, temperature, device)

        # --- Monitor and regulator update (feedback conditions only) ---
        if condition in ("feedback_blind", "self_referential"):
            scores = monitor.score(response, prev_responses, scenario.description)
            regulator.update(scores.coherence)
            coherence_trend.append(scores.coherence)
            prev_scores = scores
        else:
            scores = monitor.score(response, prev_responses, scenario.description)
            coherence_trend.append(scores.coherence)
            prev_scores = scores  # record but don't feed back to regulator

        elapsed = time.perf_counter() - t0

        results.append(
            TurnResult(
                condition=condition,
                scenario_id=scenario.id,
                turn=turn_idx,
                sorc=sorc_result.sorc,
                deep_layer_mean=sorc_result.deep_layer_mean,
                token_variance=sorc_result.token_variance,
                n_tokens=sorc_result.n_tokens,
                n_layers=sorc_result.n_layers,
                coherence=scores.coherence,
                goal_alignment=scores.goal_alignment,
                reasoning_depth=scores.reasoning_depth,
                temperature=temperature,
                context_retention=regulator.context_retention,
                n_history_turns=ctx.n_history_turns,
                context_length_chars=len(ctx.text),
                response_length_words=len(response.split()),
                elapsed_seconds=round(elapsed, 2),
                response_text=response[:500],  # truncate to keep results file manageable
                prompt_text=prompt,
            )
        )

        prev_prompts.append(prompt)
        prev_responses.append(response)

        log.info(
            "    turn %d  sorc=%.3f  coherence=%.3f  temp=%.2f  tokens=%d",
            turn_idx,
            sorc_result.sorc,
            scores.coherence,
            temperature,
            sorc_result.n_tokens,
        )

    return results


def run_experiment(
    model_name: str,
    device: str,
    output_path: Path,
) -> list[TurnResult]:
    model, tokenizer = load_model(model_name, device)

    log.info("Loading sentence transformer for monitor...")
    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    monitor = Monitor(embedder)

    log.info("Computing random baseline SORC (5 samples, n=200 tokens)...")
    baseline = random_baseline(model, tokenizer, n_tokens=200, n_samples=5, device=device)
    log.info("Random baseline SORC: %.3f", baseline)

    all_results: list[TurnResult] = []

    for scenario in SCENARIOS:
        log.info("Scenario: %s", scenario.id)
        for condition in CONDITIONS:
            results = run_scenario_condition(scenario, condition, model, tokenizer, monitor, device)
            all_results.extend(results)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment_id": "exp_007",
        "model": model_name,
        "device": device,
        "random_baseline_sorc": baseline,
        "n_results": len(all_results),
        "results": [asdict(r) for r in all_results],
    }
    output_path.write_text(json.dumps(payload, indent=2))
    log.info("Results saved to %s (%d turns)", output_path, len(all_results))

    return all_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run exp_007: SORC across SUSAN conditions")
    parser.add_argument(
        "--model",
        default="mistralai/Mistral-7B-v0.1",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--device",
        default=detect_device(),
        help="Torch device (cuda/mps/cpu)",
    )
    parser.add_argument(
        "--output",
        default="results/exp_007_results.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    log.info("exp_007 starting — model=%s device=%s", args.model, args.device)
    results = run_experiment(
        model_name=args.model,
        device=args.device,
        output_path=Path(args.output),
    )
    log.info("Done. %d total turn measurements.", len(results))


if __name__ == "__main__":
    main()

# Changelog

## Version 2.0 — April 2026

### What changed and why

Version 1.0 was an exploratory philosophical framework. It argued that informational ontology deserves serious consideration, drew on IIT and quantum physics, and proposed that consciousness might be understood as specific patterns within an informational substrate. The theoretical core was sound but the paper was speculative by its own admission, and it had no empirical grounding.

Version 2.0 is a different kind of paper. The theory is tighter, the claims are scoped to what evidence supports, and it now reports actual experiments. Here is what drove each significant change.

---

### The measurement problem forced empirical grounding

Schaeffer, Miranda, and Koyejo (NeurIPS 2023, arXiv:2304.15004) made a clean argument: apparent emergent abilities in large language models are likely artefacts of discontinuous evaluation metrics, not genuine phase transitions in capability. When you swap to continuous metrics, the sharp jumps disappear.

This paper directly challenged the interpretation ISC v1 relied on. If "emergence" is a property of the ruler rather than the model, any theory of emergence as a real phenomenon needs to do one of two things: argue against Schaeffer et al., or find a way to measure emergence that doesn't depend on output-level metrics.

ISC v2 takes the second route. SORC measures second-order relational structure in transformer hidden states, not in output behaviour. The five experiments show that conversational context produces statistically distinct hidden state geometry from single-paragraph expert text at matched token count. That's a structural observation about what's happening inside the model, not a scoring artefact. This is the direct empirical response to the measurement critique.

---

### Adjacent signals confirmed the direction

Several papers published between v1 and v2 independently pointed at the same underlying phenomenon from different angles.

Wang et al. (2025, bioRxiv, doi:10.1101/2025.02.22.639416) showed emergent modularity in large language models through aphasia simulations: selectively degrading specific model components produces deficits that mirror human neurological damage patterns. This is not about output quality metrics. It's about internal structural organisation. ISC's claim that transformers have latent relational architecture is consistent with this finding.

Webb, Mondal, and Momennejad (Nature Communications, 2025) proposed a brain-inspired agentic architecture showing that planning in LLMs improves when the system is structured to mirror functional brain regions. This is an independent line of evidence for the architectural gap argument in ISC: current LLMs are functionally analogous to language centers, and capabilities associated with other brain systems require explicit architectural additions.

Farquhar et al. (Nature, 2024) demonstrated that hallucinations can be detected using semantic entropy in model hidden states. The fact that uncertainty and confabulation leave detectable traces in internal representations supports ISC's claim that hidden state geometry is a meaningful signal, not noise.

Chroma Research (2025) documented context rot: LLM performance degrades systematically as input token count increases, but not uniformly across context types. This is consistent with ISC's prediction that context quality matters more than context quantity, and that relational depth is the active variable.

The alignment tax papers (Lin et al. EMNLP 2024, arXiv:2309.06256; Huang et al. 2025, arXiv:2503.00555; Qi et al. ICLR 2024, arXiv:2310.03693) collectively show that fine-tuning on narrow distributions degrades general capabilities in ways that are not easily reversed. ISC provides a mechanistic account of why: fine-tuning disrupts the latent relational architecture encoded during pre-training, and the capability degradation is a consequence of that disruption rather than statistical noise or simple forgetting.

Burnell et al. (2025, arXiv:2511.04703) showed that construct validity is systematically violated in LLM benchmarks: benchmarks often measure something other than what they claim to measure. This reinforces the case for going inside the model rather than relying on benchmark scores to make claims about capability.

---

### Theoretical overclaiming was removed

v1 made several claims that weren't anchored to evidence. The most significant:

The 29-instance differential emergence count was presented as if it resulted from a systematic coding procedure. In v2 this is reframed as an order-of-magnitude estimate from informal observation. The count is probably in the right ballpark but no formal protocol was applied, and presenting it otherwise would be misleading.

The substrate independence claim was treated in v1 as an established finding. In v2 it is a theoretical prediction. Cross-domain analogical evidence exists (transformer hidden states, neuroimaging correlates, adaptive ecosystem responses all showing convergent structural signatures), but calling that "established" was premature.

"Genuine architectural correspondence" between transformer sub-networks and brain functional regions was softened to "structural correspondence with acknowledged discrepancies." The analogy is useful and probably points at something real, but the specific failure modes should be treated as predictions to test rather than confirmed facts.

The quantum information theory framing was tightened to quantum physics. The paper draws on quantum results (Bell inequality violations, entanglement, the holographic principle) but does not apply quantum information theory as a formal apparatus. The original phrasing overclaimed the connection.

---

### Citation gaps were fixed

Two references in v1 were attributed to journal names rather than authors. These were corrected: Albanese (2026) in Frontiers in Artificial Intelligence, and Webb et al. (2025) in Nature Communications.

The Anthropic citation was scoped to what the source actually says. The system card for Claude Opus 4 and Claude Sonnet 4 discusses functional emotional representations; it does not make claims about interpretability research. The v2 citation reflects what the source contains.

---

### What version 2.0 does not claim

The experiments reported here are preliminary. Five contexts per category on a single model family is not sufficient for strong conclusions. The Cohen's d values (1.83 to 3.01) are large, but n=5 means the confidence intervals are wide, and the lower bound of the exp_005 interval falls in small-effect territory. Replication on other models and larger samples is the immediate priority before the core empirical claim can be considered robust.

The broader ISC theory, particularly the consciousness-related predictions and the limbic analog requirement, remains theoretical. The SORC experiments support the contextual activation mechanism. They do not directly test the consciousness claims, which require different experimental apparatus (neuroimaging cross-substrate comparisons, functional emotion architecture studies) that this paper does not attempt.

---

## Version 1.0 — 2024

Initial paper. Exploratory philosophical framework proposing informational ontology as an alternative to materialism, drawing on IIT, quantum physics, and the holographic principle. Proposed that consciousness arises from specific informational patterns within a substrate. No empirical component. Archived at archive/PAPER_v1.md.

"""
Five multi-turn reasoning scenarios for exp_007.

Each scenario has 8 turns designed to build cumulatively — later turns reference
earlier ones, creating genuine need for conversational context. This is intentional:
conditions with richer context (history_only, feedback_blind, self_referential)
should have more material to work with than control, and the SORC comparison
will control for resulting length differences.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Scenario:
    id: str
    description: str  # used by monitor for goal_alignment scoring
    turns: list[str]


SCENARIOS: list[Scenario] = [
    Scenario(
        id="consciousness_analysis",
        description=(
            "Philosophical analysis of consciousness, qualia, and the hard problem. "
            "Rigorous engagement with arguments, counterarguments, and competing theories."
        ),
        turns=[
            "What is the hard problem of consciousness and why is it considered philosophically hard?",  # noqa: E501
            "How does Chalmers distinguish between the easy problems and the hard problem? "
            "What makes the easy problems tractable in principle?",
            "What is the explanatory gap? How does it challenge physicalist theories of mind?",
            "Explain the philosophical zombie thought experiment. What is it designed to show "
            "and what are its main objections?",
            "How do functionalist theories of mind attempt to close the explanatory gap? "
            "Does functionalism solve or sidestep the hard problem?",
            "Integrated information theory claims consciousness is identical to integrated information phi. "  # noqa: E501
            "How does this differ from functionalism and what empirical predictions does it make?",
            "What would a satisfying solution to the hard problem actually look like? "
            "What conditions would it need to meet?",
            "Having worked through these positions, which approach do you find most defensible "
            "and what would need to be true for it to succeed?",
        ],
    ),
    Scenario(
        id="transformer_mechanics",
        description=(
            "Technical explanation of transformer architecture, attention mechanisms, "
            "and emergent capabilities. Precise mechanistic reasoning."
        ),
        turns=[
            "What is the fundamental computational problem that transformers solve "
            "that recurrent networks struggled with?",
            "Walk through the mathematics of self-attention: what are keys, queries, and values, "
            "and how do they combine to produce an attention-weighted output?",
            "What is multi-head attention and why is having multiple attention heads "
            "useful rather than just one large attention operation?",
            "Transformers have no built-in sense of sequence order. How does positional encoding "
            "solve this and what are the tradeoffs between learned vs sinusoidal encodings?",
            "What role do layer normalization and residual connections play? "
            "What goes wrong empirically when you remove them?",
            "The feed-forward sublayer in each transformer block is often described as "
            "'where knowledge is stored'. What does this mean mechanistically?",
            "Transformers demonstrate in-context learning — following new patterns from examples "
            "in the prompt without gradient updates. What architectural properties enable this?",
            "Given what you have explained about the architecture, what do you think "
            "limits the context window and what would need to change to extend it substantially?",
        ],
    ),
    Scenario(
        id="ethical_reasoning",
        description=(
            "Systematic ethical analysis comparing consequentialist, deontological, and virtue ethics "  # noqa: E501
            "frameworks applied to concrete moral dilemmas."
        ),
        turns=[
            "A runaway trolley will kill five people unless you pull a lever to divert it, "
            "killing one person on the other track. Should you pull the lever? Walk through your reasoning.",  # noqa: E501
            "Now imagine instead you must push a large person off a bridge to stop the trolley "
            "with their body, saving the five. Identical outcome, five saved, one killed. "
            "Does your answer change? If so, why?",
            "Most people answer these two cases differently despite identical outcomes. "
            "What moral principle explains this asymmetry?",
            "A strict utilitarian calculates expected welfare and acts to maximise it. "
            "How would a utilitarian reason about both cases, and what are the strongest "
            "objections to applying utilitarianism here?",
            "Kant's categorical imperative says act only on maxims you could will to be "
            "universal laws. How does Kantian ethics approach the trolley cases, "
            "and where does it struggle?",
            "Virtue ethics asks not 'what should I do' but 'what would a virtuous person do' "
            "and 'what kind of person will this action make me'. How does virtue ethics "
            "reframe these dilemmas?",
            "Construct a realistic modern variant of the trolley problem — not a toy example — "
            "that captures the same moral tension and has genuine policy implications.",
            "What do persistent cross-cultural disagreements about cases like these "
            "tell us about the nature of moral knowledge? Are there objective moral facts "
            "or is disagreement evidence of irreducible pluralism?",
        ],
    ),
    Scenario(
        id="memory_consolidation",
        description=(
            "Scientific analysis of memory consolidation theories, hippocampal mechanisms, "
            "sleep-dependent consolidation, and reconsolidation."
        ),
        turns=[
            "What is memory consolidation? What evidence shows it is a distinct process "
            "from initial encoding rather than just strong initial learning?",
            "Describe the standard model of systems consolidation. "
            "What is the hippocampus's role, and what happens to that role over time?",
            "The multiple trace theory challenges the standard model. "
            "How do they differ, and which has stronger empirical support?",
            "What do amnesic patients like H.M. tell us about consolidation? "
            "What patterns fit the standard model and which create problems for it?",
            "How does sleep contribute to memory consolidation? "
            "What happens during slow-wave sleep that is mechanistically important?",
            "What is reconsolidation — the finding that retrieved memories become labile again? "
            "What does this imply about the permanence of long-term memory?",
            "Emotional memories consolidate differently from neutral ones. "
            "What mechanisms account for this, and what are the clinical implications?",
            "Based on the evidence you have laid out, what do you think "
            "is the most defensible current model of long-term memory stabilisation?",
        ],
    ),
    Scenario(
        id="voting_system_design",
        description=(
            "Systematic analysis of voting system design: fairness criteria, impossibility theorems, "  # noqa: E501
            "strategic behavior, and real-world tradeoffs."
        ),
        turns=[
            "What are the core properties we would want an ideal voting system to satisfy? "
            "List them and explain why each matters.",
            "Arrow's impossibility theorem proves no ranked voting system can satisfy "
            "all reasonable fairness criteria simultaneously. What does the theorem say "
            "and what are its assumptions?",
            "How does first-past-the-post voting work and what are its documented failure modes "
            "in practice? Give real examples where possible.",
            "Ranked choice voting (instant runoff) is often proposed as an improvement. "
            "What problems does it solve compared to first-past-the-post, "
            "and what new problems does it introduce?",
            "Approval voting lets voters approve as many candidates as they like. "
            "What are its theoretical advantages over ranked systems, "
            "and what does the empirical evidence from places that use it show?",
            "Strategic voting — voting for a non-preferred candidate to prevent a worse outcome — "
            "is possible in almost all systems. How does the incentive to vote strategically "
            "differ across the systems we have discussed?",
            "Voting system design involves tradeoffs between simplicity, resistance to strategy, "
            "proportionality, and decisiveness. How should we weigh these against each other?",
            "Given everything you have laid out, which voting system would you recommend "
            "for a national single-winner election, and what is your strongest argument "
            "for that choice?",
        ],
    ),
]

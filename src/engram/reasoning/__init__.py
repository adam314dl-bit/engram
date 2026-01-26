"""Reasoning layer for Engram."""

from engram.reasoning.episode_manager import (
    EpisodeManager,
    create_episode,
)
from engram.reasoning.pipeline import (
    ReasoningPipeline,
    ReasoningResult,
    reason,
)
from engram.reasoning.re_reasoning import (
    ReReasoner,
    ReReasoningResult,
    re_reason,
)
from engram.reasoning.synthesizer import (
    Behavior,
    ResponseSynthesizer,
    SynthesisResult,
    extract_behavior,
    synthesize_response,
)

__all__ = [
    # Synthesizer
    "ResponseSynthesizer",
    "SynthesisResult",
    "Behavior",
    "extract_behavior",
    "synthesize_response",
    # Episode manager
    "EpisodeManager",
    "create_episode",
    # Re-reasoning
    "ReReasoner",
    "ReReasoningResult",
    "re_reason",
    # Pipeline
    "ReasoningPipeline",
    "ReasoningResult",
    "reason",
]

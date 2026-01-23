"""Reasoning layer for Engram.

Includes v4 agentic components:
- Intent classification
- Self-RAG validation
- Hallucination detection
- Citations
- Confidence calibration
- IRCoT multi-hop reasoning
- Async research mode
- Agentic pipeline
"""

from engram.reasoning.episode_manager import (
    EpisodeManager,
    create_episode,
)
from engram.reasoning.pipeline import (
    ReasoningPipeline,
    ReasoningResult,
    reason,
    reason_with_documents,
)
from engram.reasoning.re_reasoning import (
    ReReasoner,
    ReReasoningResult,
    re_reason,
)
from engram.reasoning.selector import (
    MemorySelector,
    SelectionResult,
    select_memories,
)
from engram.reasoning.synthesizer import (
    Behavior,
    ResponseSynthesizer,
    SynthesisResult,
    extract_behavior,
    synthesize_response,
)

# v4 Agentic components
from engram.reasoning.intent_classifier import (
    IntentClassifier,
    IntentResult,
    QueryComplexity,
    RetrievalDecision,
    classify_intent,
)
from engram.reasoning.self_rag import (
    SelfRAGResult,
    SelfRAGValidator,
    SupportLevel,
    ValidationResult,
    validate_response,
)
from engram.reasoning.hallucination_detector import (
    ClaimStatus,
    ClaimVerification,
    HallucinationDetector,
    HallucinationResult,
    detect_hallucinations,
)
from engram.reasoning.citations import (
    Citation,
    CitationManager,
    CitedResponse,
    add_citations,
)
from engram.reasoning.confidence import (
    ConfidenceCalibrator,
    ConfidenceLevel,
    ConfidenceResult,
    ResponseAction,
    calibrate_confidence,
)
from engram.reasoning.ircot import (
    IRCoTReasoner,
    IRCoTResult,
    ReasoningStep,
    ircot_reason,
)
from engram.reasoning.research_agent import (
    ResearchAgent,
    ResearchProgress,
    ResearchResult,
    ResearchStatus,
    research,
)
from engram.reasoning.agentic_pipeline import (
    AgenticMetadata,
    AgenticPipeline,
    AgenticResult,
    agentic_reason,
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
    # Selector (two-phase retrieval)
    "MemorySelector",
    "SelectionResult",
    "select_memories",
    # Pipeline
    "ReasoningPipeline",
    "ReasoningResult",
    "reason",
    "reason_with_documents",
    # v4: Intent Classification
    "IntentClassifier",
    "IntentResult",
    "QueryComplexity",
    "RetrievalDecision",
    "classify_intent",
    # v4: Self-RAG
    "SelfRAGResult",
    "SelfRAGValidator",
    "SupportLevel",
    "ValidationResult",
    "validate_response",
    # v4: Hallucination Detection
    "ClaimStatus",
    "ClaimVerification",
    "HallucinationDetector",
    "HallucinationResult",
    "detect_hallucinations",
    # v4: Citations
    "Citation",
    "CitationManager",
    "CitedResponse",
    "add_citations",
    # v4: Confidence
    "ConfidenceCalibrator",
    "ConfidenceLevel",
    "ConfidenceResult",
    "ResponseAction",
    "calibrate_confidence",
    # v4: IRCoT
    "IRCoTReasoner",
    "IRCoTResult",
    "ReasoningStep",
    "ircot_reason",
    # v4: Research Agent
    "ResearchAgent",
    "ResearchProgress",
    "ResearchResult",
    "ResearchStatus",
    "research",
    # v4: Agentic Pipeline
    "AgenticMetadata",
    "AgenticPipeline",
    "AgenticResult",
    "agentic_reason",
]

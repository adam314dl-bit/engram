"""Learning layer for Engram."""

from engram.learning.consolidation import (
    Consolidator,
    ConsolidationResult,
    maybe_consolidate,
)
from engram.learning.feedback_handler import (
    FeedbackHandler,
    FeedbackResult,
    FeedbackType,
    handle_feedback,
)
from engram.learning.hebbian import (
    create_or_strengthen_link,
    strengthen_concept_links,
    update_concept_activation,
    weaken_concept_links,
)
from engram.learning.memory_strength import (
    batch_strengthen_memories,
    batch_weaken_memories,
    strengthen_memory,
    update_memory_strength,
    weaken_memory,
)
from engram.learning.reflection import (
    Reflection,
    ReflectionResult,
    Reflector,
    maybe_reflect,
)

__all__ = [
    # Memory strength (SM-2)
    "update_memory_strength",
    "strengthen_memory",
    "weaken_memory",
    "batch_strengthen_memories",
    "batch_weaken_memories",
    # Hebbian learning
    "strengthen_concept_links",
    "weaken_concept_links",
    "update_concept_activation",
    "create_or_strengthen_link",
    # Consolidation
    "Consolidator",
    "ConsolidationResult",
    "maybe_consolidate",
    # Reflection
    "Reflector",
    "Reflection",
    "ReflectionResult",
    "maybe_reflect",
    # Feedback handler
    "FeedbackHandler",
    "FeedbackResult",
    "FeedbackType",
    "handle_feedback",
]

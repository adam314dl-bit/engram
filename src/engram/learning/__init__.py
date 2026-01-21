"""Learning layer for Engram."""

from engram.learning.consolidation import (
    Consolidator,
    ConsolidationResult,
    maybe_consolidate,
)
from engram.learning.contradiction import (
    ContradictionResult,
    ResolutionResult,
    batch_check_contradictions,
    check_and_resolve,
    detect_contradiction,
    find_potential_contradictions,
    resolve_contradiction,
)
from engram.learning.feedback_handler import (
    FeedbackHandler,
    FeedbackResult,
    FeedbackType,
    handle_feedback,
)
from engram.learning.forgetting import (
    STATUS_ACTIVE,
    STATUS_ARCHIVED,
    STATUS_DEPRIORITIZED,
    ActivationInfo,
    batch_update_activations,
    compute_base_level_activation,
    compute_retrieval_probability,
    determine_memory_status,
    get_memories_by_status,
    restore_memory,
    run_forgetting_cycle,
    update_memory_activation,
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
    # Forgetting (ACT-R)
    "ActivationInfo",
    "compute_base_level_activation",
    "compute_retrieval_probability",
    "determine_memory_status",
    "update_memory_activation",
    "batch_update_activations",
    "restore_memory",
    "get_memories_by_status",
    "run_forgetting_cycle",
    "STATUS_ACTIVE",
    "STATUS_DEPRIORITIZED",
    "STATUS_ARCHIVED",
    # Contradiction detection
    "ContradictionResult",
    "ResolutionResult",
    "detect_contradiction",
    "resolve_contradiction",
    "check_and_resolve",
    "find_potential_contradictions",
    "batch_check_contradictions",
]

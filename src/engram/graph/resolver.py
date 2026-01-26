"""Concept resolver for alias→canonical mappings.

Provides efficient concept name resolution during ingestion,
preventing duplicates by resolving to existing canonical concepts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jellyfish

from engram.preprocessing.transliteration import to_latin

if TYPE_CHECKING:
    from engram.models import Concept
    from engram.storage.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


@dataclass
class ResolvedConcept:
    """Result of concept resolution."""

    concept_id: str
    concept_name: str
    matched_by: str  # exact, alias, phonetic, none
    similarity: float  # 1.0 for exact, lower for fuzzy matches


class ConceptResolver:
    """Resolves concept names to canonical concepts.

    Maintains an in-memory cache of alias→canonical mappings
    for fast resolution during ingestion.
    """

    def __init__(
        self,
        db: "Neo4jClient",
        phonetic_threshold: float = 0.90,
    ) -> None:
        self.db = db
        self.phonetic_threshold = phonetic_threshold

        # In-memory caches
        self._alias_cache: dict[str, str] = {}  # alias (lowercase) -> canonical_id
        self._name_cache: dict[str, str] = {}  # name (lowercase) -> concept_id
        self._phonetic_cache: dict[str, str] = {}  # latin_name -> concept_id
        self._loaded = False

    async def load_alias_cache(self) -> None:
        """Load alias→canonical mappings from database."""
        if self._loaded:
            return

        logger.info("Loading concept alias cache...")

        # Load all active concepts with their aliases
        query = """
        MATCH (c:Concept)
        WHERE c.status IS NULL OR c.status = 'active'
        RETURN c.id as id, c.name as name, c.aliases as aliases
        """
        results = await self.db.execute_query(query)

        for r in results:
            concept_id = r["id"]
            name = r["name"].lower()
            aliases = r.get("aliases") or []

            # Add main name to caches
            self._name_cache[name] = concept_id
            self._phonetic_cache[to_latin(name)] = concept_id

            # Add aliases
            for alias in aliases:
                alias_lower = alias.lower()
                self._alias_cache[alias_lower] = concept_id
                self._phonetic_cache[to_latin(alias_lower)] = concept_id

        self._loaded = True
        logger.info(
            f"Loaded {len(self._name_cache)} concepts, "
            f"{len(self._alias_cache)} aliases"
        )

    async def resolve_concept(
        self,
        name: str,
        use_phonetic: bool = True,
    ) -> ResolvedConcept | None:
        """Resolve a concept name to its canonical concept.

        Args:
            name: Concept name to resolve
            use_phonetic: Whether to use phonetic matching

        Returns:
            ResolvedConcept if found, None otherwise
        """
        await self.load_alias_cache()

        name_lower = name.lower()

        # 1. Exact name match
        if name_lower in self._name_cache:
            concept_id = self._name_cache[name_lower]
            return ResolvedConcept(
                concept_id=concept_id,
                concept_name=name,
                matched_by="exact",
                similarity=1.0,
            )

        # 2. Alias match
        if name_lower in self._alias_cache:
            concept_id = self._alias_cache[name_lower]
            return ResolvedConcept(
                concept_id=concept_id,
                concept_name=name,
                matched_by="alias",
                similarity=1.0,
            )

        # 3. Phonetic match (if enabled)
        if use_phonetic:
            latin_name = to_latin(name_lower)

            # Exact phonetic match
            if latin_name in self._phonetic_cache:
                concept_id = self._phonetic_cache[latin_name]
                return ResolvedConcept(
                    concept_id=concept_id,
                    concept_name=name,
                    matched_by="phonetic",
                    similarity=1.0,
                )

            # Fuzzy phonetic match
            best_match: tuple[str, float] | None = None
            for cached_latin, concept_id in self._phonetic_cache.items():
                similarity = jellyfish.jaro_winkler_similarity(latin_name, cached_latin)
                if similarity >= self.phonetic_threshold:
                    if best_match is None or similarity > best_match[1]:
                        best_match = (concept_id, similarity)

            if best_match:
                return ResolvedConcept(
                    concept_id=best_match[0],
                    concept_name=name,
                    matched_by="phonetic",
                    similarity=best_match[1],
                )

        return None

    async def resolve_concepts_batch(
        self,
        names: list[str],
        use_phonetic: bool = True,
    ) -> dict[str, ResolvedConcept | None]:
        """Resolve multiple concept names efficiently.

        Args:
            names: Concept names to resolve
            use_phonetic: Whether to use phonetic matching

        Returns:
            Mapping of name -> ResolvedConcept (or None)
        """
        await self.load_alias_cache()

        results: dict[str, ResolvedConcept | None] = {}
        for name in names:
            results[name] = await self.resolve_concept(name, use_phonetic)
        return results

    def add_to_cache(
        self,
        concept_id: str,
        name: str,
        aliases: list[str] | None = None,
    ) -> None:
        """Add a new concept to the cache.

        Called after creating a new concept during ingestion.

        Args:
            concept_id: Concept ID
            name: Concept name
            aliases: Optional aliases
        """
        name_lower = name.lower()
        self._name_cache[name_lower] = concept_id
        self._phonetic_cache[to_latin(name_lower)] = concept_id

        for alias in aliases or []:
            alias_lower = alias.lower()
            self._alias_cache[alias_lower] = concept_id
            self._phonetic_cache[to_latin(alias_lower)] = concept_id

    def clear_cache(self) -> None:
        """Clear all caches (e.g., after database reset)."""
        self._alias_cache.clear()
        self._name_cache.clear()
        self._phonetic_cache.clear()
        self._loaded = False


# Global resolver instance
_resolver: ConceptResolver | None = None


async def get_concept_resolver(db: "Neo4jClient") -> ConceptResolver:
    """Get or create the global concept resolver."""
    global _resolver
    if _resolver is None:
        _resolver = ConceptResolver(db)
        await _resolver.load_alias_cache()
    return _resolver


def clear_concept_resolver() -> None:
    """Clear the global concept resolver."""
    global _resolver
    if _resolver is not None:
        _resolver.clear_cache()
    _resolver = None

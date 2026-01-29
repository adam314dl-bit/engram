#!/usr/bin/env python
"""Debug ingestion pipeline in dry-run mode.

Processes a document through all ingestion stages WITHOUT saving to Neo4j,
showing what would be extracted at each phase to help debug data loss.

Usage:
    uv run python scripts/debug_ingestion.py /path/to/document.md
    uv run python scripts/debug_ingestion.py /path/to/document.md --compare
    uv run python scripts/debug_ingestion.py /path/to/document.md -v
    uv run python scripts/debug_ingestion.py /path/to/document.md -o report.json
    uv run python scripts/debug_ingestion.py /path/to/document.md --skip-llm

Examples:
    # Basic usage - show all extraction stages
    uv run python scripts/debug_ingestion.py tests/fixtures/sample_doc.md

    # Compare with existing data in Neo4j
    uv run python scripts/debug_ingestion.py /path/to/document.md --compare

    # Verbose mode - include LLM prompts and raw responses
    uv run python scripts/debug_ingestion.py /path/to/document.md -v

    # Output to file
    uv run python scripts/debug_ingestion.py /path/to/document.md -o report.json

    # Skip LLM-based phases (parsing, tables, lists, persons only)
    uv run python scripts/debug_ingestion.py /path/to/document.md --skip-llm
"""

# Disable telemetry but allow model downloads
import os
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import argparse
import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    # Reduce noise
    logging.getLogger("neo4j").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


@dataclass
class StageResult:
    """Result from a single extraction stage."""
    name: str
    input_count: int
    output_count: int
    items: list[Any] = field(default_factory=list)
    dropped: list[Any] = field(default_factory=list)
    llm_calls: int = 0
    duration_ms: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class DebugReport:
    """Full debug report for a document."""
    file_path: str
    document_id: str
    document_title: str
    content_length: int
    stages: list[StageResult] = field(default_factory=list)
    total_concepts: int = 0
    total_memories: int = 0
    total_relations: int = 0
    total_llm_calls: int = 0
    total_duration_ms: float = 0.0


@dataclass
class ComparisonResult:
    """Comparison of extracted data with Neo4j."""
    doc_exists: bool
    existing_doc_id: str | None = None
    concepts_new: list[str] = field(default_factory=list)
    concepts_matched: list[str] = field(default_factory=list)
    memories_new: int = 0
    memories_matched: int = 0
    memories_missing: int = 0


def format_duration(ms: float) -> str:
    """Format duration in human-readable form."""
    if ms >= 1000:
        return f"{ms/1000:.2f}s"
    return f"{ms:.0f}ms"


def print_header(title: str, char: str = "=") -> None:
    """Print a section header."""
    print(f"\n{char*70}")
    print(f"  {title}")
    print(f"{char*70}")


def print_subheader(title: str) -> None:
    """Print a subsection header."""
    print(f"\n  {title}")
    print(f"  {'-' * len(title)}")


class IngestionDebugger:
    """Runs ingestion pipeline in dry-run mode, collecting data at each stage."""

    def __init__(self, db=None, verbose: bool = False, skip_llm: bool = False, skip_relations: bool = False):
        self.db = db  # Optional, for --compare mode
        self.verbose = verbose
        self.skip_llm = skip_llm
        self.skip_relations = skip_relations
        self.stages: list[StageResult] = []

    async def debug_document(self, file_path: str) -> DebugReport:
        """Process document through all stages without saving."""
        from engram.ingestion.parser import DocumentParser
        from engram.preprocessing.table_parser import extract_tables, remove_tables_from_text
        from engram.preprocessing.list_parser import extract_lists
        from engram.ingestion.person_extractor import PersonExtractor
        from engram.ingestion.concept_extractor import ConceptExtractor
        from engram.ingestion.memory_extractor import MemoryExtractor
        from engram.ingestion.relationship_extractor import RelationshipExtractor
        from engram.preprocessing.content_context import (
            ContentContextExtractor,
            extract_page_metadata_from_content,
        )
        from engram.retrieval.quality_filter import calculate_chunk_quality, is_quality_chunk
        from engram.config import settings

        total_start = time.perf_counter()
        total_llm_calls = 0

        # Initialize extractors
        parser = DocumentParser()
        context_extractor = ContentContextExtractor()
        person_extractor = PersonExtractor()

        # LLM-based extractors (only initialize if not skipping)
        concept_extractor = None
        memory_extractor = None
        relationship_extractor = None
        table_enricher = None
        list_enricher = None

        if not self.skip_llm:
            concept_extractor = ConceptExtractor()
            memory_extractor = MemoryExtractor(concept_extractor=concept_extractor)
            relationship_extractor = RelationshipExtractor()

            # Lazy import enrichers
            from engram.preprocessing.table_enricher import TableEnricher
            from engram.preprocessing.list_enricher import ListEnricher
            table_enricher = TableEnricher()
            list_enricher = ListEnricher()

        # === PHASE 0: Document Parsing ===
        print_header("PHASE 0: DOCUMENT PARSING")
        phase_start = time.perf_counter()

        path = Path(file_path)
        document = await parser.parse_file(path)

        # Extract page metadata
        page_metadata = extract_page_metadata_from_content(document.content)
        if not page_metadata.title:
            page_metadata.title = document.title
        if not page_metadata.url and document.source_path:
            page_metadata.url = document.source_path

        phase_duration = (time.perf_counter() - phase_start) * 1000

        print(f"  File: {file_path}")
        print(f"  Title: {document.title}")
        print(f"  Source URL: {document.source_path or 'N/A'}")
        print(f"  Content length: {len(document.content):,} chars")
        if page_metadata.space_name:
            hierarchy = " > ".join(
                filter(None, [page_metadata.space_name] + page_metadata.parent_pages + [page_metadata.title])
            )
            print(f"  Page hierarchy: {hierarchy}")

        self.stages.append(StageResult(
            name="parsing",
            input_count=1,
            output_count=1,
            duration_ms=phase_duration,
            details={
                "title": document.title,
                "source_path": document.source_path,
                "content_length": len(document.content),
                "space_name": page_metadata.space_name,
                "parent_pages": page_metadata.parent_pages,
            }
        ))

        # === PHASE 1: Table Extraction ===
        print_header("PHASE 1: TABLE EXTRACTION")
        phase_start = time.perf_counter()

        tables = extract_tables(document.content)
        table_memories = []

        print(f"  Found {len(tables)} tables")

        if tables:
            for i, table in enumerate(tables, 1):
                print_subheader(f"Table {i} [lines {table.start_line}-{table.end_line}]")
                print(f"    Headers: {' | '.join(table.headers[:5])}" +
                      ("..." if len(table.headers) > 5 else ""))
                print(f"    Rows: {len(table.rows)}")
                print(f"    Is comparison: {'Yes' if table.is_comparison_table else 'No'}")
                if table.context:
                    print(f"    Context: {table.context[:100]}...")

                # Enrich table (requires LLM)
                if not self.skip_llm and table_enricher:
                    from engram.preprocessing.table_enricher import (
                        create_fact_memories_from_table,
                        create_memory_from_table,
                    )

                    context = context_extractor.extract_context(
                        document.content,
                        table.start_line,
                        table.end_line,
                        page_metadata,
                    )

                    try:
                        enriched = await table_enricher.enrich_v41(table, context)
                        total_llm_calls += 1

                        print(f"\n    -> Enriched summary: {enriched.description[:80]}...")
                        print(f"    -> Key facts extracted: {len(enriched.key_facts)}")

                        # Create memories
                        table_memory = create_memory_from_table(enriched, doc_id=document.id)
                        fact_memories = create_fact_memories_from_table(enriched, doc_id=document.id)

                        table_memories.append(table_memory)
                        table_memories.extend(fact_memories)

                        print(f"    -> Memories created: {1 + len(fact_memories)} (1 table + {len(fact_memories)} facts)")

                        if self.verbose and enriched.key_facts:
                            print("    -> Facts:")
                            for fact in enriched.key_facts[:3]:
                                print(f"       - {fact[:60]}...")
                    except Exception as e:
                        print(f"    -> ERROR: {e}")
                else:
                    print("    -> (skipped LLM enrichment)")

        phase_duration = (time.perf_counter() - phase_start) * 1000

        self.stages.append(StageResult(
            name="tables",
            input_count=len(tables),
            output_count=len(table_memories),
            llm_calls=len(tables),
            duration_ms=phase_duration,
            items=[{"headers": t.headers, "rows": len(t.rows)} for t in tables],
        ))

        # Remove tables from content for text processing
        text_content = remove_tables_from_text(document.content) if tables else document.content

        # === PHASE 1.5: List Extraction ===
        print_header("PHASE 1.5: LIST EXTRACTION")
        phase_start = time.perf_counter()

        parsed_lists = extract_lists(text_content)
        list_memories = []

        print(f"  Found {len(parsed_lists)} lists")

        if parsed_lists:
            for i, parsed_list in enumerate(parsed_lists, 1):
                print_subheader(f"List {i} [lines {parsed_list.start_line}-{parsed_list.end_line}] - Type: {parsed_list.list_type}")
                print(f"    Items: {len(parsed_list.items)}")
                if parsed_list.context:
                    print(f"    Context: {parsed_list.context[:100]}...")

                # Enrich list (requires LLM)
                if not self.skip_llm and list_enricher:
                    from engram.preprocessing.list_enricher import (
                        create_fact_memories_from_list,
                        create_memory_from_list,
                    )

                    context = context_extractor.extract_context(
                        text_content,
                        parsed_list.start_line,
                        parsed_list.end_line,
                        page_metadata,
                    )

                    try:
                        enriched = await list_enricher.enrich(parsed_list, context)
                        total_llm_calls += 1

                        print(f"\n    -> Enriched description: {enriched.description[:80]}...")
                        print(f"    -> Key facts: {len(enriched.key_facts)}")

                        # Create memories
                        list_memory = create_memory_from_list(enriched, doc_id=document.id)
                        fact_memories = create_fact_memories_from_list(enriched, doc_id=document.id)

                        list_memories.append(list_memory)
                        list_memories.extend(fact_memories)

                        print(f"    -> Memories created: {1 + len(fact_memories)} (1 list + {len(fact_memories)} facts)")
                    except Exception as e:
                        print(f"    -> ERROR: {e}")
                else:
                    print("    -> (skipped LLM enrichment)")

        phase_duration = (time.perf_counter() - phase_start) * 1000

        self.stages.append(StageResult(
            name="lists",
            input_count=len(parsed_lists),
            output_count=len(list_memories),
            llm_calls=len(parsed_lists),
            duration_ms=phase_duration,
            items=[{"type": lst.list_type, "items": len(lst.items)} for lst in parsed_lists],
        ))

        # === PHASE 1.6: Person Extraction ===
        print_header("PHASE 1.6: PERSON EXTRACTION")
        phase_start = time.perf_counter()

        person_result = person_extractor.extract(text_content, document.id)
        person_memories = []

        print(f"  Found {len(person_result.persons)} persons")

        from engram.ingestion.parser import generate_id
        from engram.models import SemanticMemory

        for person in person_result.persons:
            print(f"\n  Person: {person.canonical_name}")
            if person.role:
                print(f"    Role: {person.role}")
            if person.team:
                print(f"    Team: {person.team}")
            print(f"    Source: Natasha NER + PyMorphy3")

            # Create memory for person with role
            if person.role:
                content = f"{person.canonical_name} — {person.role}"
                if person.team:
                    content += f" (команда {person.team})"

                person_memories.append(SemanticMemory(
                    id=generate_id(),
                    content=content,
                    concept_ids=[],
                    source_doc_ids=[document.id],
                    memory_type="fact",
                    importance=6.0,
                    metadata={
                        "source_type": "person",
                        "person_name": person.canonical_name,
                        "person_role": person.role,
                        "person_team": person.team,
                    },
                ))

        phase_duration = (time.perf_counter() - phase_start) * 1000

        self.stages.append(StageResult(
            name="persons",
            input_count=len(person_result.persons),
            output_count=len(person_memories),
            llm_calls=0,  # No LLM, uses Natasha/regex
            duration_ms=phase_duration,
            items=[{"name": p.canonical_name, "role": p.role, "team": p.team} for p in person_result.persons],
        ))

        # === PHASE 2: Concept Extraction ===
        print_header("PHASE 2: CONCEPT EXTRACTION")
        phase_start = time.perf_counter()

        # Placeholder for LLM extraction results
        concept_result_concepts = []
        memory_result_memories = []
        memory_result_concepts = []

        if not self.skip_llm and concept_extractor:
            concept_result = await concept_extractor.extract(text_content)
            concept_result_concepts = concept_result.concepts
            total_llm_calls += 1

            print(f"  Extracted {len(concept_result_concepts)} concepts")

            for i, concept in enumerate(concept_result_concepts, 1):
                desc = concept.description[:50] + "..." if concept.description and len(concept.description) > 50 else (concept.description or "")
                print(f"  {i:2}. {concept.name} ({concept.type}) - {desc}")
        else:
            print("  (skipped - requires LLM)")

        phase_duration = (time.perf_counter() - phase_start) * 1000

        self.stages.append(StageResult(
            name="concepts",
            input_count=1,
            output_count=len(concept_result_concepts),
            llm_calls=1 if not self.skip_llm else 0,
            duration_ms=phase_duration,
            items=[{"name": c.name, "type": c.type, "description": c.description} for c in concept_result_concepts],
        ))

        # === PHASE 2.1: Memory Extraction ===
        print_header("PHASE 2.1: MEMORY EXTRACTION")
        phase_start = time.perf_counter()

        if not self.skip_llm and memory_extractor:
            memory_result = await memory_extractor.extract(
                text_content,
                document.title,
                document.id,
            )
            memory_result_memories = memory_result.memories
            memory_result_concepts = memory_result.concepts
            total_llm_calls += 1

            print(f"  Extracted {len(memory_result_memories)} memories")

            for i, memory in enumerate(memory_result_memories, 1):
                content_preview = memory.content[:60] + "..." if len(memory.content) > 60 else memory.content
                concepts_str = ", ".join(memory.concept_ids[:3]) + ("..." if len(memory.concept_ids) > 3 else "")
                print(f"\n  Memory {i} [{memory.memory_type}, importance={memory.importance}]:")
                print(f"    Content: {content_preview}")
                if memory.search_content and memory.search_content != memory.content:
                    search_preview = memory.search_content[:60] + "..." if len(memory.search_content) > 60 else memory.search_content
                    print(f"    Search: {search_preview}")
                if memory.concept_ids:
                    print(f"    Concepts: [{concepts_str}]")
        else:
            print("  (skipped - requires LLM)")

        phase_duration = (time.perf_counter() - phase_start) * 1000

        self.stages.append(StageResult(
            name="memories",
            input_count=1,
            output_count=len(memory_result_memories),
            llm_calls=1 if not self.skip_llm else 0,
            duration_ms=phase_duration,
            items=[{
                "id": m.id,
                "type": m.memory_type,
                "importance": m.importance,
                "content": m.content[:100],
                "concepts": m.concept_ids,
            } for m in memory_result_memories],
        ))

        # Merge concepts from memory extraction
        all_concepts: dict[str, Any] = {}
        for c in concept_result_concepts:
            all_concepts[c.name] = c
        for c in memory_result_concepts:
            if c.name not in all_concepts:
                all_concepts[c.name] = c
        concepts = list(all_concepts.values())

        # Combine all memories
        all_memories = memory_result_memories + table_memories + list_memories + person_memories

        # === PHASE 2.5: Quality Filtering ===
        print_header("PHASE 2.5: QUALITY FILTERING")
        phase_start = time.perf_counter()

        kept_memories = []
        dropped_memories = []

        for memory in all_memories:
            source_type = memory.metadata.get("source_type") if memory.metadata else None

            # Keep enriched content without filtering
            if source_type in ("table", "list", "person", "table_fact", "list_fact"):
                kept_memories.append(memory)
            elif is_quality_chunk(memory.content, settings.chunk_quality_threshold, settings.min_chunk_words):
                kept_memories.append(memory)
            else:
                score = calculate_chunk_quality(memory.content)
                dropped_memories.append({
                    "content": memory.content[:50],
                    "score": score.overall,
                    "reason": f"score={score.overall:.2f} < threshold={settings.chunk_quality_threshold}",
                })

        phase_duration = (time.perf_counter() - phase_start) * 1000

        print(f"  Kept: {len(kept_memories)} memories")
        print(f"  Dropped: {len(dropped_memories)} memories (low quality)")

        for dm in dropped_memories:
            print(f"    - \"{dm['content']}...\" ({dm['reason']})")

        self.stages.append(StageResult(
            name="quality_filter",
            input_count=len(all_memories),
            output_count=len(kept_memories),
            dropped=dropped_memories,
            duration_ms=phase_duration,
        ))

        # === PHASE 2.6: Relation Extraction ===
        print_header("PHASE 2.6: RELATION EXTRACTION")
        phase_start = time.perf_counter()

        relations = []
        if self.skip_relations:
            print("  (skipped - --skip-relations flag)")
        elif not self.skip_llm and relationship_extractor and len(concepts) >= 2:
            try:
                relations = await relationship_extractor.extract_relationships(
                    document.content,
                    concepts,
                )
                total_llm_calls += 1

                print(f"  Extracted {len(relations)} relations")

                for i, rel in enumerate(relations, 1):
                    source_name = next((c.name for c in concepts if c.id == rel.source_id), rel.source_id[:8])
                    target_name = next((c.name for c in concepts if c.id == rel.target_id), rel.target_id[:8])
                    print(f"  {i:2}. {source_name} --[{rel.relation_type}]--> {target_name} (weight: {rel.weight:.2f})")
            except Exception as e:
                print(f"  ERROR: {e}")
                print("  (relation extraction failed, continuing...)")
        elif self.skip_llm:
            print("  (skipped - requires LLM)")
        else:
            print("  Extracted 0 relations (need at least 2 concepts)")

        phase_duration = (time.perf_counter() - phase_start) * 1000

        self.stages.append(StageResult(
            name="relations",
            input_count=len(concepts),
            output_count=len(relations),
            llm_calls=1 if relations and not self.skip_llm else 0,
            duration_ms=phase_duration,
            items=[{
                "source": rel.source_id,
                "target": rel.target_id,
                "type": rel.relation_type,
                "weight": rel.weight,
            } for rel in relations],
        ))

        # === SUMMARY ===
        total_duration = (time.perf_counter() - total_start) * 1000

        print_header("SUMMARY")

        # Count memories by type
        memory_counts = {
            "table": len(table_memories),
            "list": len(list_memories),
            "person": len(person_memories),
            "text": len(memory_result_memories),
        }

        memory_breakdown = ", ".join(f"{v} {k}" for k, v in memory_counts.items() if v > 0)

        print(f"  Total concepts: {len(concepts)}")
        print(f"  Total memories: {len(kept_memories)} ({memory_breakdown})")
        print(f"  Total relations: {len(relations)}")
        if self.skip_llm:
            print(f"  LLM calls: 0 (--skip-llm mode)")
        else:
            print(f"  LLM calls: {total_llm_calls} (1 concepts + 1 memories + {len(tables)} tables + {len(parsed_lists)} lists" +
                  f"{' + 1 relations' if relations else ''})")
        print(f"  Total time: {format_duration(total_duration)}")

        return DebugReport(
            file_path=str(file_path),
            document_id=document.id,
            document_title=document.title,
            content_length=len(document.content),
            stages=self.stages,
            total_concepts=len(concepts),
            total_memories=len(kept_memories),
            total_relations=len(relations),
            total_llm_calls=total_llm_calls,
            total_duration_ms=total_duration,
        )

    async def compare_with_neo4j(self, report: DebugReport) -> ComparisonResult:
        """Compare extracted data with existing Neo4j data."""
        if not self.db:
            return ComparisonResult(doc_exists=False)

        print_header("COMPARISON WITH NEO4J (--compare)")

        # Find existing document by source hash
        from engram.ingestion.parser import compute_hash

        path = Path(report.file_path)
        try:
            async with open(path) as f:
                content = await f.read()
            source_hash = compute_hash(content)
        except Exception:
            print("  Could not read file to compute hash")
            return ComparisonResult(doc_exists=False)

        # Query Neo4j for existing document
        query = """
        MATCH (d:Document {source_hash: $hash})
        RETURN d.id as id, d.title as title
        LIMIT 1
        """
        result = await self.db.query(query, {"hash": source_hash})

        if not result:
            print("  Document not found in Neo4j (new document)")
            return ComparisonResult(doc_exists=False)

        existing_doc_id = result[0]["id"]
        print(f"  Document already ingested: Yes ({existing_doc_id[:12]}...)")

        # Get existing concepts linked to this document
        concept_query = """
        MATCH (d:Document {id: $doc_id})<-[:FROM_DOCUMENT]-(m:SemanticMemory)-[:RELATES_TO]->(c:Concept)
        RETURN DISTINCT c.name as name
        """
        existing_concepts = await self.db.query(concept_query, {"doc_id": existing_doc_id})
        existing_concept_names = {r["name"].lower() for r in existing_concepts}

        # Compare concepts
        extracted_concept_names = set()
        for stage in report.stages:
            if stage.name == "concepts":
                for item in stage.items:
                    extracted_concept_names.add(item["name"].lower())

        new_concepts = extracted_concept_names - existing_concept_names
        matched_concepts = extracted_concept_names & existing_concept_names

        print(f"\n  Concepts:")
        print(f"    + NEW: {len(new_concepts)} concepts not in Neo4j")
        print(f"    = MATCH: {len(matched_concepts)} concepts already exist")

        if new_concepts:
            for name in list(new_concepts)[:5]:
                print(f"      + {name}")
            if len(new_concepts) > 5:
                print(f"      ... and {len(new_concepts) - 5} more")

        # Get existing memory count
        memory_query = """
        MATCH (d:Document {id: $doc_id})<-[:FROM_DOCUMENT]-(m:SemanticMemory)
        RETURN count(m) as count
        """
        memory_result = await self.db.query(memory_query, {"doc_id": existing_doc_id})
        existing_memory_count = memory_result[0]["count"] if memory_result else 0

        print(f"\n  Memories:")
        print(f"    Extracted now: {report.total_memories}")
        print(f"    In Neo4j: {existing_memory_count}")

        return ComparisonResult(
            doc_exists=True,
            existing_doc_id=existing_doc_id,
            concepts_new=list(new_concepts),
            concepts_matched=list(matched_concepts),
            memories_new=max(0, report.total_memories - existing_memory_count),
            memories_matched=min(report.total_memories, existing_memory_count),
            memories_missing=max(0, existing_memory_count - report.total_memories),
        )


async def main(args: argparse.Namespace) -> None:
    """Run ingestion debug."""
    setup_logging(args.verbose)

    # Check file exists
    file_path = Path(args.file_path)
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    if not file_path.suffix.lower() in (".md", ".txt", ".markdown"):
        print(f"Error: Unsupported file type: {file_path.suffix}")
        print("Supported: .md, .txt, .markdown")
        sys.exit(1)

    # Connect to Neo4j if --compare
    db = None
    if args.compare:
        from engram.storage.neo4j_client import get_client
        db = await get_client()

    # Run debug
    debugger = IngestionDebugger(
        db=db,
        verbose=args.verbose,
        skip_llm=args.skip_llm,
        skip_relations=args.skip_relations,
    )
    report = await debugger.debug_document(str(file_path))

    # Compare with Neo4j if requested
    comparison = None
    if args.compare and db:
        comparison = await debugger.compare_with_neo4j(report)

    # Output to file if requested
    if args.output:
        output_data = {
            "report": asdict(report),
            "comparison": asdict(comparison) if comparison else None,
        }
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n  Report saved to: {output_path}")

    # Cleanup
    if db:
        await db.close()

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Debug ingestion pipeline in dry-run mode (no Neo4j writes)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "file_path",
        help="Path to document file (.md, .txt)",
    )

    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare with existing data in Neo4j",
    )

    parser.add_argument(
        "-o", "--output",
        help="Output report to JSON file",
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output with additional details",
    )

    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip LLM-based phases (parsing, tables, lists, persons only)",
    )

    parser.add_argument(
        "--skip-relations",
        action="store_true",
        help="Skip relation extraction (requires BGE-M3 model download)",
    )

    args = parser.parse_args()
    asyncio.run(main(args))

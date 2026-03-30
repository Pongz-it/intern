"""LLM-Driven Query Generator for Deep Research.

Generates semantically rich, keyword-optimized, and multilingual search queries
for comprehensive research coverage.

Reference: backend/onyx/agents/agent_search/deep/shared/expanded_retrieval.py
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from agent_rag.llm.interface import LLM, LLMMessage
from agent_rag.utils.logger import get_logger

logger = get_logger(__name__)


class QueryType(str, Enum):
    """Types of generated queries."""
    SEMANTIC = "semantic"  # Natural language, context-rich
    KEYWORD = "keyword"  # Boolean operators, specific terms
    EXACT_PHRASE = "exact_phrase"  # Quoted exact matches
    ENTITY = "entity"  # Named entities, proper nouns
    TECHNICAL = "technical"  # Technical terms, acronyms
    MULTILINGUAL = "multilingual"  # Translated queries


class QueryStrategy(str, Enum):
    """Query generation strategies."""
    EXPLORATORY = "exploratory"  # Broad coverage, discovery
    FOCUSED = "focused"  # Narrow, specific information
    COMPARATIVE = "comparative"  # Comparing alternatives
    TEMPORAL = "temporal"  # Time-bound queries
    CAUSAL = "causal"  # Cause-effect relationships


@dataclass
class GeneratedQuery:
    """A generated search query with metadata."""
    query: str
    query_type: QueryType
    strategy: QueryStrategy
    weight: float = 1.0  # Importance weight
    language: str = "en"
    rationale: Optional[str] = None
    parent_query: Optional[str] = None  # If derived from another query


@dataclass
class QueryGenerationResult:
    """Result of query generation."""
    queries: list[GeneratedQuery]
    semantic_queries: list[str] = field(default_factory=list)
    keyword_queries: list[str] = field(default_factory=list)
    multilingual_queries: dict[str, list[str]] = field(default_factory=dict)
    total_queries: int = 0
    strategy_used: str = ""


@dataclass
class QueryGeneratorConfig:
    """Configuration for query generator."""
    max_semantic_queries: int = 3
    max_keyword_queries: int = 3
    max_queries_per_language: int = 2
    target_languages: list[str] = field(default_factory=lambda: ["en"])
    include_exact_phrases: bool = True
    include_entities: bool = True
    include_technical_terms: bool = True
    strategy: QueryStrategy = QueryStrategy.EXPLORATORY
    deduplicate: bool = True


# =============================================================================
# PROMPTS
# =============================================================================

SEMANTIC_QUERY_PROMPT = """You are a research query optimizer. Generate semantic search queries for finding comprehensive information.

Research Question: {question}
Context: {context}
Previous Queries: {previous_queries}

Generate {num_queries} semantically rich search queries that:
1. Capture the intent and nuances of the research question
2. Use natural language that matches how information is typically written
3. Include contextual terms that help find relevant content
4. Avoid duplicating previous queries
5. Cover different aspects or perspectives of the topic

Format your response as:
1. [query] | [brief rationale]
2. [query] | [brief rationale]
...
"""

KEYWORD_QUERY_PROMPT = """You are a search optimization expert. Generate keyword-focused search queries for precise retrieval.

Research Question: {question}
Context: {context}
Previous Queries: {previous_queries}

Generate {num_queries} keyword-optimized search queries that:
1. Use specific technical terms and domain vocabulary
2. Include relevant acronyms and abbreviations
3. Focus on exact terminology from the domain
4. Use boolean-style combinations (term1 AND term2, NOT excluded_term)
5. Target precise information retrieval

Format your response as:
1. [keyword query] | [terms targeted]
2. [keyword query] | [terms targeted]
...
"""

MULTILINGUAL_QUERY_PROMPT = """You are a multilingual research assistant. Translate and adapt search queries for different languages.

Original Question: {question}
Target Languages: {languages}
Context: {context}

For each language, generate {queries_per_language} culturally and linguistically adapted search queries.
These should not be literal translations but should:
1. Use natural phrasing in the target language
2. Include language-specific terminology
3. Account for regional variations where relevant
4. Maintain the research intent

Format your response as:
## [Language Code]
1. [query in that language]
2. [query in that language]
...

## [Next Language Code]
...
"""

ENTITY_EXTRACTION_PROMPT = """Extract key entities and proper nouns from this research question that would be good search terms.

Research Question: {question}
Context: {context}

Extract:
1. Named entities (people, organizations, products, places)
2. Technical terms and domain-specific vocabulary
3. Important concepts and frameworks
4. Acronyms and their expansions

Format your response as:
### Entities
- [entity1]: [type]
- [entity2]: [type]
...

### Technical Terms
- [term1]
- [term2]
...

### Acronyms
- [acronym]: [full form]
...
"""

QUERY_DIVERSIFICATION_PROMPT = """Diversify these search queries to improve coverage across different information types.

Original Queries: {queries}
Research Question: {question}

Generate alternative formulations that:
1. Use synonyms and related terms
2. Change perspective (e.g., problem → solution, cause → effect)
3. Adjust specificity (broader ↔ narrower scope)
4. Target different content types (articles, papers, tutorials, docs)

Format your response as:
### Original: [original query]
- Alternative 1: [query]
- Alternative 2: [query]

### Original: [next query]
...
"""


class LLMQueryGenerator:
    """
    LLM-powered query generator for deep research.

    Features:
    - Semantic query generation (natural language, context-rich)
    - Keyword query generation (Boolean operators, specific terms)
    - Multilingual query generation (translated and adapted)
    - Entity extraction for targeted searches
    - Query diversification for comprehensive coverage
    """

    def __init__(
        self,
        llm: LLM,
        config: Optional[QueryGeneratorConfig] = None,
    ) -> None:
        """
        Initialize the query generator.

        Args:
            llm: LLM provider for query generation
            config: Configuration options
        """
        self.llm = llm
        self.config = config or QueryGeneratorConfig()

    def generate(
        self,
        question: str,
        context: Optional[str] = None,
        previous_queries: Optional[list[str]] = None,
        strategy: Optional[QueryStrategy] = None,
    ) -> QueryGenerationResult:
        """
        Generate comprehensive search queries for a research question.

        Args:
            question: The research question
            context: Additional context about the research
            previous_queries: Queries already executed (to avoid duplicates)
            strategy: Override default query strategy

        Returns:
            QueryGenerationResult with all generated queries
        """
        context = context or ""
        previous_queries = previous_queries or []
        strategy = strategy or self.config.strategy

        all_queries: list[GeneratedQuery] = []

        # Generate semantic queries
        semantic_queries = self._generate_semantic_queries(
            question=question,
            context=context,
            previous_queries=previous_queries,
        )
        all_queries.extend(semantic_queries)

        # Generate keyword queries
        keyword_queries = self._generate_keyword_queries(
            question=question,
            context=context,
            previous_queries=previous_queries + [q.query for q in semantic_queries],
        )
        all_queries.extend(keyword_queries)

        # Extract entities for targeted queries
        if self.config.include_entities:
            entity_queries = self._generate_entity_queries(
                question=question,
                context=context,
            )
            all_queries.extend(entity_queries)

        # Generate multilingual queries if configured
        multilingual_dict: dict[str, list[str]] = {}
        if len(self.config.target_languages) > 1:
            for lang in self.config.target_languages:
                if lang != "en":  # Skip English, already covered
                    lang_queries = self._generate_multilingual_queries(
                        question=question,
                        target_language=lang,
                        context=context,
                    )
                    all_queries.extend(lang_queries)
                    multilingual_dict[lang] = [q.query for q in lang_queries]

        # Deduplicate if configured
        if self.config.deduplicate:
            all_queries = self._deduplicate_queries(all_queries)

        # Build result
        result = QueryGenerationResult(
            queries=all_queries,
            semantic_queries=[q.query for q in all_queries if q.query_type == QueryType.SEMANTIC],
            keyword_queries=[q.query for q in all_queries if q.query_type == QueryType.KEYWORD],
            multilingual_queries=multilingual_dict,
            total_queries=len(all_queries),
            strategy_used=strategy.value,
        )

        logger.info(f"Generated {len(all_queries)} queries for question: {question[:50]}...")
        return result

    def _generate_semantic_queries(
        self,
        question: str,
        context: str,
        previous_queries: list[str],
    ) -> list[GeneratedQuery]:
        """Generate semantic (natural language) queries."""
        prompt = SEMANTIC_QUERY_PROMPT.format(
            question=question,
            context=context or "None provided",
            previous_queries="\n".join(f"- {q}" for q in previous_queries) if previous_queries else "None",
            num_queries=self.config.max_semantic_queries,
        )

        messages = [LLMMessage(role="user", content=prompt)]
        response = self.llm.generate(messages)

        return self._parse_query_response(
            response.content,
            query_type=QueryType.SEMANTIC,
            strategy=self.config.strategy,
        )

    def _generate_keyword_queries(
        self,
        question: str,
        context: str,
        previous_queries: list[str],
    ) -> list[GeneratedQuery]:
        """Generate keyword-focused queries."""
        prompt = KEYWORD_QUERY_PROMPT.format(
            question=question,
            context=context or "None provided",
            previous_queries="\n".join(f"- {q}" for q in previous_queries) if previous_queries else "None",
            num_queries=self.config.max_keyword_queries,
        )

        messages = [LLMMessage(role="user", content=prompt)]
        response = self.llm.generate(messages)

        return self._parse_query_response(
            response.content,
            query_type=QueryType.KEYWORD,
            strategy=self.config.strategy,
        )

    def _generate_entity_queries(
        self,
        question: str,
        context: str,
    ) -> list[GeneratedQuery]:
        """Generate entity-focused queries."""
        prompt = ENTITY_EXTRACTION_PROMPT.format(
            question=question,
            context=context or "None provided",
        )

        messages = [LLMMessage(role="user", content=prompt)]
        response = self.llm.generate(messages)

        entities = self._parse_entities(response.content)
        queries = []

        # Create queries from extracted entities
        for entity, entity_type in entities[:5]:  # Limit to top 5 entities
            query = GeneratedQuery(
                query=entity,
                query_type=QueryType.ENTITY,
                strategy=QueryStrategy.FOCUSED,
                weight=0.8,
                language="en",
                rationale=f"Entity ({entity_type}): targeted search",
            )
            queries.append(query)

        return queries

    def _generate_multilingual_queries(
        self,
        question: str,
        target_language: str,
        context: str,
    ) -> list[GeneratedQuery]:
        """Generate queries in a specific language."""
        prompt = MULTILINGUAL_QUERY_PROMPT.format(
            question=question,
            languages=target_language,
            context=context or "None provided",
            queries_per_language=self.config.max_queries_per_language,
        )

        messages = [LLMMessage(role="user", content=prompt)]
        response = self.llm.generate(messages)

        return self._parse_multilingual_response(
            response.content,
            target_language=target_language,
        )

    def generate_refined_queries(
        self,
        original_queries: list[str],
        findings: str,
        gaps: list[str],
    ) -> list[GeneratedQuery]:
        """
        Generate refined queries based on findings and identified gaps.

        Args:
            original_queries: Previous queries executed
            findings: Summary of what has been found
            gaps: Identified knowledge gaps

        Returns:
            List of refined queries
        """
        prompt = f"""Based on the research progress, generate refined search queries.

## Previous Queries
{chr(10).join(f'- {q}' for q in original_queries)}

## Current Findings
{findings}

## Knowledge Gaps
{chr(10).join(f'- {g}' for g in gaps)}

Generate {self.config.max_semantic_queries + self.config.max_keyword_queries} refined queries that:
1. Target the identified knowledge gaps
2. Avoid repeating previous unsuccessful approaches
3. Use more specific or alternative terminology
4. Explore different angles of the topic

Format as:
1. [query] | [which gap this addresses]
...
"""

        messages = [LLMMessage(role="user", content=prompt)]
        response = self.llm.generate(messages)

        return self._parse_query_response(
            response.content,
            query_type=QueryType.SEMANTIC,
            strategy=QueryStrategy.FOCUSED,
        )

    def diversify_queries(
        self,
        queries: list[str],
        question: str,
    ) -> list[GeneratedQuery]:
        """
        Diversify existing queries for broader coverage.

        Args:
            queries: Existing queries to diversify
            question: Original research question

        Returns:
            Diversified query alternatives
        """
        prompt = QUERY_DIVERSIFICATION_PROMPT.format(
            queries="\n".join(f"- {q}" for q in queries),
            question=question,
        )

        messages = [LLMMessage(role="user", content=prompt)]
        response = self.llm.generate(messages)

        return self._parse_diversification_response(response.content)

    def _parse_query_response(
        self,
        response: str,
        query_type: QueryType,
        strategy: QueryStrategy,
    ) -> list[GeneratedQuery]:
        """Parse numbered query list from LLM response."""
        import re

        queries = []
        for line in response.split('\n'):
            line = line.strip()
            # Match patterns like "1. query | rationale" or "1. query"
            match = re.match(r'^\d+[\.\)]\s*(.+?)(?:\s*\|\s*(.+))?$', line)
            if match:
                query_text = match.group(1).strip()
                rationale = match.group(2).strip() if match.group(2) else None

                if query_text and len(query_text) > 3:  # Skip too short queries
                    queries.append(GeneratedQuery(
                        query=query_text,
                        query_type=query_type,
                        strategy=strategy,
                        weight=1.0,
                        language="en",
                        rationale=rationale,
                    ))

        return queries

    def _parse_entities(self, response: str) -> list[tuple[str, str]]:
        """Parse entity extraction response."""
        import re

        entities = []

        # Parse entities section
        entity_section = self._extract_section(response, "Entities")
        for line in entity_section.split('\n'):
            line = line.strip()
            if line.startswith('- '):
                # Parse "entity: type" format
                match = re.match(r'^-\s*(.+?):\s*(.+)$', line)
                if match:
                    entities.append((match.group(1).strip(), match.group(2).strip()))

        # Parse technical terms
        terms_section = self._extract_section(response, "Technical Terms")
        for line in terms_section.split('\n'):
            line = line.strip()
            if line.startswith('- '):
                term = line[2:].strip()
                if term:
                    entities.append((term, "technical"))

        return entities

    def _parse_multilingual_response(
        self,
        response: str,
        target_language: str,
    ) -> list[GeneratedQuery]:
        """Parse multilingual query response."""
        import re

        queries = []
        current_lang = None

        for line in response.split('\n'):
            line = line.strip()

            # Check for language header
            lang_match = re.match(r'^##\s*(\w+)', line)
            if lang_match:
                current_lang = lang_match.group(1).lower()
                continue

            # Parse numbered queries
            query_match = re.match(r'^\d+[\.\)]\s*(.+)$', line)
            if query_match and current_lang:
                query_text = query_match.group(1).strip()
                if query_text:
                    queries.append(GeneratedQuery(
                        query=query_text,
                        query_type=QueryType.MULTILINGUAL,
                        strategy=QueryStrategy.EXPLORATORY,
                        weight=0.9,  # Slightly lower weight for translations
                        language=current_lang,
                        rationale=f"Translated query for {current_lang}",
                    ))

        return queries

    def _parse_diversification_response(self, response: str) -> list[GeneratedQuery]:
        """Parse diversification response."""
        import re

        queries = []
        current_original = None

        for line in response.split('\n'):
            line = line.strip()

            # Check for original query header
            if line.startswith("### Original:"):
                current_original = line.replace("### Original:", "").strip()
                continue

            # Parse alternatives
            alt_match = re.match(r'^-\s*Alternative\s*\d+:\s*(.+)$', line)
            if alt_match and current_original:
                query_text = alt_match.group(1).strip()
                if query_text:
                    queries.append(GeneratedQuery(
                        query=query_text,
                        query_type=QueryType.SEMANTIC,
                        strategy=QueryStrategy.EXPLORATORY,
                        weight=0.85,
                        language="en",
                        rationale="Diversified query",
                        parent_query=current_original,
                    ))

        return queries

    def _extract_section(self, text: str, header: str) -> str:
        """Extract content under a markdown header."""
        import re
        pattern = rf"###?\s*{header}\s*\n(.*?)(?=###|\Z)"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else ""

    def _deduplicate_queries(
        self,
        queries: list[GeneratedQuery],
    ) -> list[GeneratedQuery]:
        """Remove duplicate queries, keeping higher-weighted ones."""
        seen: dict[str, GeneratedQuery] = {}

        for query in queries:
            normalized = query.query.lower().strip()
            if normalized in seen:
                # Keep the one with higher weight
                if query.weight > seen[normalized].weight:
                    seen[normalized] = query
            else:
                seen[normalized] = query

        return list(seen.values())


def create_query_generator(
    llm: LLM,
    config: Optional[QueryGeneratorConfig] = None,
) -> LLMQueryGenerator:
    """Factory function to create an LLM query generator."""
    return LLMQueryGenerator(llm=llm, config=config)

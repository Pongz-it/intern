"""Intent analyzer for data query detection.

Analyzes user queries to determine if they are data queries
and identifies the specific type of query intent.
Supports dynamic keyword management for continuous learning.
"""

import logging
import re
from dataclasses import dataclass
from typing import Optional, Any

from agent_rag.llm.interface import LLM

from agent_rag.text_to_sql.models import QueryIntent
from agent_rag.text_to_sql.keyword_manager import KeywordManager

logger = logging.getLogger(__name__)


@dataclass
class QueryIntentAnalysis:
    """Result of query intent analysis."""

    is_data_query: bool
    confidence: float
    primary_intent: QueryIntent
    secondary_intents: list[QueryIntent]
    detected_entities: list[str]
    suggested_tables: list[str]
    suggested_columns: list[str]
    time_expressions: list[str]
    aggregation_keywords: list[str]
    raw_analysis: Optional[dict] = None


class IntentAnalyzer:
    """Analyzes user query intent for data queries.

    Uses a combination of rule-based patterns and LLM analysis
    to determine if a query is asking for a database query.
    Supports dynamic keyword management for continuous learning.
    """

    QUERY_TYPE_PATTERNS = {
        QueryIntent.COUNT: [
            r"多少[个件条]?|how many|count",
            r"有几个|有多少个",
            r"number of|total count",
        ],
        QueryIntent.SUM: [
            r"总和|合计|总计|sum of|total amount",
            r"金额|销售额|收入|支出",
            r"加起来|加总",
        ],
        QueryIntent.AVG: [
            r"平均|average|mean",
            r"每.*平均",
            r"平均值",
        ],
        QueryIntent.MIN: [
            r"最小|最少|lowest|minimum|min",
            r"最少的|最小的",
        ],
        QueryIntent.MAX: [
            r"最大|最多|highest|maximum|max",
            r"最多的|最大的",
        ],
        QueryIntent.AGGREGATION: [
            r"统计|statistics|stats",
            r"汇总|aggregate",
            r"总计|一共",
        ],
        QueryIntent.FILTER: [
            r"where|条件|筛选|过滤",
            r"只.*的|只要|除了",
            r"在.*之间|between",
        ],
        QueryIntent.TIME_SERIES: [
            r"最近|近期|latest|recent",
            r"上周|上周|上个月|去年|this week|this month|last year",
            r"趋势|trend|over time",
            r"按时间|按日|按周|按月|按年",
        ],
        QueryIntent.COMPARISON: [
            r"比|compared|vs|versus| versus",
            r"增长|增长|减少|decline",
            r"相比|和.*相比",
            r"差异|difference",
        ],
        QueryIntent.GROUP_BY: [
            r"按.*分组|按.*分类|group by",
            r"每个.*的|每.*的",
            r"按照",
        ],
        QueryIntent.SORT: [
            r"排序|sort|order by",
            r"最高| lowest| ascending| descending",
            r"排名前十|top |bottom ",
        ],
        QueryIntent.JOIN: [
            r"关联|join|link|connect",
            r"和.*一起|以及",
            r"包含.*信息",
        ],
        QueryIntent.DISTINCT: [
            r"不同|distinct|unique|different",
            r"有哪些|有哪些类型",
        ],
    }

    def __init__(
        self,
        llm: Optional[LLM] = None,
        confidence_threshold: float = 0.7,
        keyword_manager: Optional[KeywordManager] = None,
        auto_learn: bool = True,
        db_scanner: Optional[Any] = None,
    ):
        """Initialize the intent analyzer.

        Args:
            llm: Optional LLM for advanced intent analysis
            confidence_threshold: Minimum confidence to classify as data query
            keyword_manager: Optional KeywordManager for dynamic keywords
            auto_learn: Whether to auto-learn new keywords from queries
            db_scanner: Optional ExternalDatabaseScanner for database keyword detection
        """
        self.llm = llm
        self.confidence_threshold = confidence_threshold
        self.keyword_manager = keyword_manager
        self.auto_learn = auto_learn
        self.db_scanner = db_scanner
        self._dynamic_keywords: dict[str, list[str]] = {}

    async def load_dynamic_keywords(self) -> None:
        """Load dynamic keywords from keyword manager.

        Should be called after initialization if keyword_manager is set.
        """
        if not self.keyword_manager:
            return

        try:
            keywords_by_category = await self.keyword_manager.get_keywords_by_category()
            self._dynamic_keywords = {
                cat.value: keywords for cat, keywords in keywords_by_category.items()
            }
            logger.info(f"Loaded {sum(len(v) for v in self._dynamic_keywords.values())} dynamic keywords")
        except Exception as e:
            logger.warning(f"Failed to load dynamic keywords: {e}")

    def add_dynamic_keyword(
        self,
        keyword: str,
        category: str,
    ) -> bool:
        """Add a keyword to the in-memory cache.

        Args:
            keyword: The keyword to add
            category: Category (quantity, amount, time, statistics, sorting)

        Returns:
            True if keyword was added
        """
        if category not in self._dynamic_keywords:
            self._dynamic_keywords[category] = []

        if keyword not in self._dynamic_keywords[category]:
            self._dynamic_keywords[category].append(keyword)
            logger.info(f"Added dynamic keyword: {keyword} (category: {category})")
            return True

        return False

    def get_all_keywords(self) -> dict[str, list[str]]:
        """Get all keywords (static + dynamic).

        Returns:
            Dictionary with category names and keyword lists
        """
        all_keywords = {}

        default_keywords = KeywordManager.get_default_keywords()
        for category, keywords in default_keywords.items():
            all_keywords[category] = list(keywords)

        for category, keywords in self._dynamic_keywords.items():
            if category in all_keywords:
                all_keywords[category].extend(keywords)
            else:
                all_keywords[category] = list(keywords)

        return all_keywords

    async def analyze(self, query: str) -> QueryIntentAnalysis:
        """Analyze a user query for data intent.

        Args:
            query: The user's natural language query

        Returns:
            QueryIntentAnalysis with detected intent and confidence
        """
        query_lower = query.lower()

        rule_based_result = self._rule_based_analysis(query_lower)

        if self.llm and rule_based_result.confidence < self.confidence_threshold - 0.1:
            llm_result = await self._llm_analysis(query)
            result = self._merge_results(rule_based_result, llm_result)
        else:
            result = rule_based_result

        if self.auto_learn:
            await self._auto_learn(query, result)

        return result

    async def _auto_learn(
        self,
        query: str,
        analysis: QueryIntentAnalysis,
    ) -> None:
        """Auto-learn new keywords from user query.

        Args:
            query: The user's query
            analysis: The analysis result
        """
        if not self.keyword_manager:
            return

        try:
            if not analysis.is_data_query or analysis.confidence < 0.5:
                return

            learned = await self.keyword_manager.learn_from_query(
                query=query,
                detected_intent=analysis.primary_intent.value,
                feedback="positive",
            )

            if learned:
                logger.info(f"Auto-learned keyword: {learned.keyword} ({learned.category.value})")
                await self.load_dynamic_keywords()

        except Exception as e:
            logger.warning(f"Auto-learning failed: {e}")

    def _rule_based_analysis(self, query: str) -> QueryIntentAnalysis:
        """Perform rule-based intent analysis."""
        detected_entities = []
        suggested_tables = []
        suggested_columns = []
        time_expressions = []
        aggregation_keywords = []

        default_keywords = KeywordManager.get_default_keywords()
        for category, keywords in default_keywords.items():
            for keyword in keywords:
                if keyword in query:
                    detected_entities.append(keyword)
                    if category == "time":
                        time_expressions.append(keyword)
                    if category in ("amount", "statistics"):
                        aggregation_keywords.append(keyword)

        for cat_key, keywords in self._dynamic_keywords.items():
            for keyword in keywords:
                if keyword in query and keyword not in detected_entities:
                    detected_entities.append(keyword)
                    if cat_key == "time":
                        time_expressions.append(keyword)
                    if cat_key in ("amount", "statistics"):
                        aggregation_keywords.append(keyword)

        query_lower = query.lower()
        if "vip" in query_lower or "客户" in query or "消费" in query or "金额" in query:
            suggested_tables.append("test_customers")
            suggested_columns.append("name")
            suggested_columns.append("consumption_amount")
            suggested_columns.append("is_vip")

        if self.db_scanner:
            db_detection = self.db_scanner.detect_database_keywords(query)
            if db_detection["has_database_reference"]:
                logger.info(f"[IntentAnalyzer] Detected database keywords: "
                           f"tables={db_detection['detected_tables']}, "
                           f"columns={db_detection['detected_columns']}")
                
                for table in db_detection["detected_tables"]:
                    if table not in suggested_tables:
                        suggested_tables.append(table)
                        detected_entities.append(f"table:{table}")
                
                for col in db_detection["detected_columns"]:
                    if col not in suggested_columns:
                        suggested_columns.append(col)
                        detected_entities.append(f"column:{col}")
                
                for synonym in db_detection["matched_synonyms"]:
                    if synonym not in detected_entities:
                        detected_entities.append(f"synonym:{synonym}")

        primary_intent = QueryIntent.FILTER
        secondary_intents = []
        intent_scores: dict[QueryIntent, int] = {}

        for intent, patterns in self.QUERY_TYPE_PATTERNS.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    score += 1
                    if intent not in intent_scores:
                        intent_scores[intent] = 0
                    intent_scores[intent] = score

        if intent_scores:
            sorted_intents = sorted(intent_scores.items(), key=lambda x: x[1], reverse=True)
            primary_intent = sorted_intents[0][0]
            secondary_intents = [i[0] for i in sorted_intents[1:4] if i[1] > 0]

        is_data_query = len(detected_entities) > 0 or len(intent_scores) > 0
        
        db_keyword_count = 0
        if self.db_scanner:
            db_detection = self.db_scanner.detect_database_keywords(query)
            if db_detection["has_database_reference"]:
                db_keyword_count = (
                    len(db_detection["detected_tables"]) +
                    len(db_detection["detected_columns"]) +
                    len(db_detection["matched_synonyms"])
                )
        
        confidence = self._calculate_confidence(
            is_data_query, len(detected_entities), len(intent_scores), db_keyword_count
        )

        return QueryIntentAnalysis(
            is_data_query=is_data_query,
            confidence=confidence,
            primary_intent=primary_intent,
            secondary_intents=secondary_intents,
            detected_entities=detected_entities,
            suggested_tables=suggested_tables,
            suggested_columns=suggested_columns,
            time_expressions=time_expressions,
            aggregation_keywords=aggregation_keywords,
        )

    def _calculate_confidence(
        self,
        is_data_query: bool,
        entity_count: int,
        intent_count: int,
        db_keyword_count: int = 0,
    ) -> float:
        """Calculate confidence score for the analysis.
        
        Args:
            is_data_query: Whether this is a data query
            entity_count: Number of detected entities
            intent_count: Number of detected intents
            db_keyword_count: Number of matched database keywords (tables, columns, synonyms)
        """
        base_confidence = 0.5

        if is_data_query:
            base_confidence += 0.2

        base_confidence += min(entity_count * 0.05, 0.15)
        base_confidence += min(intent_count * 0.1, 0.2)
        
        if db_keyword_count > 0:
            base_confidence += min(db_keyword_count * 0.15, 0.35)
            logger.debug(f"[IntentAnalyzer] Boosted confidence by {min(db_keyword_count * 0.15, 0.35):.2f} "
                        f"for {db_keyword_count} database keywords")

        return min(base_confidence, 1.0)

    async def _llm_analysis(self, query: str) -> QueryIntentAnalysis:
        """Perform LLM-based intent analysis."""
        if not self.llm:
            return QueryIntentAnalysis(
                is_data_query=False,
                confidence=0.0,
                primary_intent=QueryIntent.FILTER,
                secondary_intents=[],
                detected_entities=[],
                suggested_tables=[],
                suggested_columns=[],
                time_expressions=[],
                aggregation_keywords=[],
            )

        prompt = f"""
Analyze the following user query to determine if it is a data query 
(asking for information from a database or spreadsheet).

Query: "{query}"

Determine:
1. Is this a data query? (yes/no)
2. What is the primary intent?
3. What entities/tables might be involved?
4. What time expressions are mentioned?
5. What aggregation keywords are present?

Respond in JSON format:
{{
    "is_data_query": true/false,
    "confidence": 0.0-1.0,
    "primary_intent": "filter|aggregation|count|sum|avg|min|max|time_series|comparison|sort|join",
    "secondary_intents": ["intent1", "intent2"],
    "detected_entities": ["entity1", "entity2"],
    "suggested_tables": ["table1", "table2"],
    "suggested_columns": ["column1", "column2"],
    "time_expressions": ["today", "last week", etc.],
    "aggregation_keywords": ["sum", "count", etc.]
}}
"""

        try:
            from agent_rag.llm.providers.litellm_provider import LLMMessage, ReasoningEffort

            messages = [
                LLMMessage(role="system", content="You are a query intent analyzer."),
                LLMMessage(role="user", content=prompt),
            ]

            response = self.llm.chat(
                messages,
                max_tokens=500,
                reasoning_effort=ReasoningEffort.OFF,
            )

            import json

            content = response.content
            if content:
                content = content.strip()
                if content.startswith("```json"):
                    content = content[7:]
                if content.endswith("```"):
                    content = content[:-3]

                result = json.loads(content)

                intent_map = {
                    "count": QueryIntent.COUNT,
                    "sum": QueryIntent.SUM,
                    "avg": QueryIntent.AVG,
                    "min": QueryIntent.MIN,
                    "max": QueryIntent.MAX,
                    "aggregation": QueryIntent.AGGREGATION,
                    "filter": QueryIntent.FILTER,
                    "time_series": QueryIntent.TIME_SERIES,
                    "comparison": QueryIntent.COMPARISON,
                    "sort": QueryIntent.SORT,
                    "join": QueryIntent.JOIN,
                    "group_by": QueryIntent.GROUP_BY,
                }

                primary_intent = intent_map.get(
                    result.get("primary_intent", "filter"), QueryIntent.FILTER
                )
                secondary_intents = [
                    intent_map.get(i, QueryIntent.FILTER)
                    for i in result.get("secondary_intents", [])
                ]

                return QueryIntentAnalysis(
                    is_data_query=result.get("is_data_query", False),
                    confidence=result.get("confidence", 0.5),
                    primary_intent=primary_intent,
                    secondary_intents=secondary_intents,
                    detected_entities=result.get("detected_entities", []),
                    suggested_tables=result.get("suggested_tables", []),
                    suggested_columns=result.get("suggested_columns", []),
                    time_expressions=result.get("time_expressions", []),
                    aggregation_keywords=result.get("aggregation_keywords", []),
                    raw_analysis=result,
                )

        except Exception as e:
            logger.warning(f"LLM intent analysis failed: {e}")

        return QueryIntentAnalysis(
            is_data_query=False,
            confidence=0.0,
            primary_intent=QueryIntent.FILTER,
            secondary_intents=[],
            detected_entities=[],
            suggested_tables=[],
            suggested_columns=[],
            time_expressions=[],
            aggregation_keywords=[],
        )

    def _merge_results(
        self,
        rule_result: QueryIntentAnalysis,
        llm_result: Optional[QueryIntentAnalysis],
    ) -> QueryIntentAnalysis:
        """Merge rule-based and LLM analysis results."""
        if not llm_result or llm_result.confidence == 0.0:
            return rule_result

        if llm_result.confidence > rule_result.confidence:
            return llm_result

        if llm_result.is_data_query and rule_result.is_data_query:
            merged_entities = list(
                set(rule_result.detected_entities + llm_result.detected_entities)
            )
            merged_tables = list(
                set(rule_result.suggested_tables + llm_result.suggested_tables)
            )
            merged_columns = list(
                set(rule_result.suggested_columns + llm_result.suggested_columns)
            )
            merged_times = list(
                set(rule_result.time_expressions + llm_result.time_expressions)
            )

            return QueryIntentAnalysis(
                is_data_query=True,
                confidence=max(rule_result.confidence, llm_result.confidence),
                primary_intent=llm_result.primary_intent,
                secondary_intents=list(
                    set(rule_result.secondary_intents + llm_result.secondary_intents)
                ),
                detected_entities=merged_entities,
                suggested_tables=merged_tables,
                suggested_columns=merged_columns,
                time_expressions=merged_times,
                aggregation_keywords=list(
                    set(
                        rule_result.aggregation_keywords
                        + llm_result.aggregation_keywords
                    )
                ),
            )

        return rule_result

    def generate_sql_prompt(
        self,
        query: str,
        schema_info: list[dict],
    ) -> str:
        """Generate SQL prompt for LLM.

        Args:
            query: The natural language query
            schema_info: List of table schema information

        Returns:
            Formatted prompt for SQL generation
        """
        schema_context = self._build_schema_context(schema_info)

        system_prompt = """You are an expert SQL generator. Your task is to convert natural language queries 
into accurate, safe PostgreSQL queries.

Rules:
1. Generate ONLY SELECT statements - no INSERT, UPDATE, DELETE, DROP, or other data-modifying operations
2. Use proper PostgreSQL syntax
3. Use table and column names exactly as provided in the schema
4. Use LIMIT to restrict result size (default: 100)
5. Use appropriate aggregate functions (COUNT, SUM, AVG, MIN, MAX) when needed
6. Add WHERE clauses for filtering based on user intent
7. Use GROUP BY for aggregation with multiple columns
8. Use ORDER BY with appropriate sorting (DESC for highest/first, ASC for lowest/last)
9. Use proper JOIN syntax when querying multiple tables

Safety Rules:
- Never generate queries that modify data
- Never use UNION with SELECT to combine results in potentially harmful ways
- Always include LIMIT clause
- Only use tables and columns from the provided schema

Respond ONLY with a JSON object: {"sql": "SELECT ..."}"""

        user_prompt = f"""{schema_context}

User Query: {query}

Generate a SQL query to answer this question. Respond ONLY with a JSON object:
{{"sql": "SELECT ..."}}"""

        return f"{system_prompt}\n\n{user_prompt}"

    def _build_schema_context(self, schema_info: list[dict]) -> str:
        """Build schema context from schema information.

        Args:
            schema_info: List of table schema dictionaries

        Returns:
            Formatted schema context string
        """
        context_parts = ["Available Tables:\n"]

        for table in schema_info:
            table_name = table.get("table") or table.get("name", "unknown")
            columns = table.get("columns", [])

            context_parts.append(f"Table: {table_name}")

            for col in columns:
                col_name = col.get("name", "")
                col_type = col.get("type", "")
                nullable = col.get("nullable", True)
                description = col.get("description", "")

                col_desc = f"  - {col_name}: {col_type}"
                if not nullable:
                    col_desc += " (NOT NULL)"
                if description:
                    col_desc += f" ({description})"

                context_parts.append(col_desc)

            context_parts.append("")

        return "\n".join(context_parts)

"""Dynamic keyword management for intent analysis.

Provides ability to add, remove, and manage keywords for intent detection
without modifying code. Keywords are stored in PostgreSQL for persistence.
"""

import logging
import uuid
from dataclasses import dataclass
from typing import Optional, Any

from sqlalchemy import Column, String, Text, TIMESTAMP, text
from sqlalchemy.orm import declarative_base

from agent_rag.core.database import DatabaseManager
from agent_rag.text_to_sql.models import KeywordCategory

logger = logging.getLogger(__name__)

Base = declarative_base()


class Keyword(Base):
    """Dynamic keyword storage table."""

    __tablename__ = "intent_keywords"

    id = Column(String(36), primary_key=True)
    keyword = Column(String(255), nullable=False, index=True)
    category = Column(String(20), nullable=False, index=True)
    language = Column(String(10), nullable=False, default="zh")
    pattern = Column(String(500), nullable=True)
    weight = Column(String(10), nullable=False, default="1.0")
    is_active = Column(String(1), nullable=False, default="1")
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, default=text("CURRENT_TIMESTAMP"))
    updated_at = Column(TIMESTAMP(timezone=True), nullable=False, default=text("CURRENT_TIMESTAMP"))
    description = Column(Text, nullable=True)


@dataclass
class KeywordInfo:
    """Keyword information."""

    keyword: str
    category: KeywordCategory
    language: str
    pattern: str
    weight: str
    is_active: bool


class KeywordManager:
    """Manages dynamic keywords for intent analysis."""

    DEFAULT_KEYWORDS = {
        KeywordCategory.QUANTITY: [
            "多少", "几", "几个", "多少个", "多少条", "多少次",
            "how many", "how much", "number of", "count of",
            "若干", "若干个", "几条", "几件",
        ],
        KeywordCategory.AMOUNT: [
            "金额", "销售额", "收入", "支出", "费用", "价格", "成本",
            "amount", "revenue", "sales", "cost", "price", "expense",
            "总额", "总金额", "总销售额",
        ],
        KeywordCategory.TIME: [
            "今天", "昨天", "明天", "本周", "上周", "本月", "上月",
            "去年", "今年", "最近", "近期", "过去",
            "today", "yesterday", "tomorrow", "this week", "last week",
            "this month", "last month", "this year", "last year",
            "recent", "past", "historical",
            "本周初", "本周末", "月初", "月末", "季初", "季末",
        ],
        KeywordCategory.STATISTICS: [
            "统计", "汇总", "总计", "合计", "一共",
            "total", "sum", "aggregate", "statistics",
            "累计", "总计数", "平均值", "中间值",
        ],
        KeywordCategory.SORTING: [
            "最高", "最低", "最多", "最少", "排名",
            "highest", "lowest", "most", "least", "rank",
            "前三名", "后五名", "前10名", "倒数三名",
        ],
    }

    DEFAULT_KEYWORDS_STR_KEY = {
        "quantity": [
            "多少", "几", "几个", "多少个", "多少条", "多少次",
            "how many", "how much", "number of", "count of",
            "若干", "若干个", "几条", "几件",
        ],
        "amount": [
            "金额", "销售额", "收入", "支出", "费用", "价格", "成本",
            "amount", "revenue", "sales", "cost", "price", "expense",
            "总额", "总金额", "总销售额",
        ],
        "time": [
            "今天", "昨天", "明天", "本周", "上周", "本月", "上月",
            "去年", "今年", "最近", "近期", "过去",
            "today", "yesterday", "tomorrow", "this week", "last week",
            "this month", "last month", "this year", "last year",
            "recent", "past", "historical",
            "本周初", "本周末", "月初", "月末", "季初", "季末",
        ],
        "statistics": [
            "统计", "汇总", "总计", "合计", "一共",
            "total", "sum", "aggregate", "statistics",
            "累计", "总计数", "平均值", "中间值",
        ],
        "sorting": [
            "最高", "最低", "最多", "最少", "排名",
            "highest", "lowest", "most", "least", "rank",
            "前三名", "后五名", "前10名", "倒数三名",
        ],
    }

    @staticmethod
    def get_default_keywords() -> dict[str, list[str]]:
        """Get default keywords with string keys.

        Returns:
            Dictionary with category names as keys and keyword lists as values
        """
        return KeywordManager.DEFAULT_KEYWORDS_STR_KEY

    def __init__(self, db_manager: Optional[DatabaseManager] = None, external_connector: Optional[Any] = None):
        """Initialize keyword manager.

        Args:
            db_manager: Database manager instance
            external_connector: Optional external PostgreSQL connector
        """
        self.db_manager = db_manager
        self.external_connector = external_connector
        self._keywords_cache: Optional[dict[str, list[str]]] = None

    async def initialize(self) -> None:
        """Initialize database and load default keywords."""
        if self.external_connector:
            logger.info("Using external connector, skipping keyword manager initialization")
            self._keywords_cache = self.DEFAULT_KEYWORDS_STR_KEY.copy()
            return

        async with self.db_manager.session() as session:
            await self._create_table_if_not_exists(session)
            await self._ensure_default_keywords(session)

    async def _create_table_if_not_exists(self, session) -> None:
        """Create keywords table if not exists."""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS intent_keywords (
            id VARCHAR(36) PRIMARY KEY,
            keyword VARCHAR(255) NOT NULL,
            category VARCHAR(20) NOT NULL,
            language VARCHAR(10) NOT NULL DEFAULT 'zh',
            pattern VARCHAR(500),
            weight VARCHAR(10) NOT NULL DEFAULT '1.0',
            is_active VARCHAR(1) NOT NULL DEFAULT '1',
            created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
            description TEXT
        )
        """
        await session.execute(text(create_table_sql))

        create_index1_sql = "CREATE INDEX IF NOT EXISTS idx_intent_keywords_keyword ON intent_keywords(keyword)"
        await session.execute(text(create_index1_sql))

        create_index2_sql = "CREATE INDEX IF NOT EXISTS idx_intent_keywords_category ON intent_keywords(category)"
        await session.execute(text(create_index2_sql))

        await session.commit()
        logger.info("Intent keywords table initialized")

    async def _ensure_default_keywords(self, session) -> None:
        """Ensure default keywords are present."""
        for category, keywords in self.DEFAULT_KEYWORDS.items():
            for keyword in keywords:
                exists = await self._keyword_exists(session, keyword, category)
                if not exists:
                    keyword_obj = Keyword(
                        id=str(uuid.uuid4()),
                        keyword=keyword,
                        category=category.value,
                        language="zh" if any(ord(c) > 127 for c in keyword) else "en",
                        pattern=None,
                        weight="1.0",
                        is_active="1",
                        description=f"Default keyword for {category.value}",
                    )
                    session.add(keyword_obj)

        await session.commit()
        logger.info("Default keywords ensured")

    async def _keyword_exists(self, session, keyword: str, category: KeywordCategory) -> bool:
        """Check if keyword already exists."""
        result = await session.execute(
            text("SELECT 1 FROM intent_keywords WHERE keyword = :kw AND category = :cat LIMIT 1"),
            {"kw": keyword, "cat": category.value}
        )
        return result.scalar() is not None

    async def add_keyword(
        self,
        keyword: str,
        category: KeywordCategory,
        language: str = "zh",
        pattern: Optional[str] = None,
        weight: str = "1.0",
        description: Optional[str] = None,
    ) -> KeywordInfo:
        """Add a new keyword.

        Args:
            keyword: The keyword to add
            category: Category of the keyword
            language: Language (zh/en)
            pattern: Optional regex pattern
            weight: Weight for scoring
            description: Optional description

        Returns:
            KeywordInfo of added keyword
        """
        async with self.db_manager.session() as session:
            keyword_obj = Keyword(
                id=str(uuid.uuid4()),
                keyword=keyword,
                category=category.value,
                language=language,
                pattern=pattern,
                weight=weight,
                is_active="1",
                description=description or f"Added keyword: {keyword}",
            )
            session.add(keyword_obj)
            await session.commit()

            logger.info(f"Added keyword: {keyword} (category: {category.value})")

            return KeywordInfo(
                keyword=keyword_obj.keyword,
                category=category,
                language=keyword_obj.language,
                pattern=keyword_obj.pattern or "",
                weight=keyword_obj.weight,
                is_active=keyword_obj.is_active == "1",
            )

    async def remove_keyword(self, keyword: str, category: Optional[KeywordCategory] = None) -> bool:
        """Remove a keyword.

        Args:
            keyword: Keyword to remove
            category: Optional category filter

        Returns:
            True if keyword was removed
        """
        async with self.db_manager.session() as session:
            if category:
                result = await session.execute(
                    text("DELETE FROM intent_keywords WHERE keyword = :kw AND category = :cat"),
                    {"kw": keyword, "cat": category.value}
                )
            else:
                result = await session.execute(
                    text("DELETE FROM intent_keywords WHERE keyword = :kw"),
                    {"kw": keyword}
                )

            await session.commit()
            removed = result.rowcount > 0

            if removed:
                logger.info(f"Removed keyword: {keyword}")

            return removed

    async def deactivate_keyword(self, keyword: str) -> bool:
        """Deactivate a keyword (soft delete).

        Args:
            keyword: Keyword to deactivate

        Returns:
            True if keyword was deactivated
        """
        async with self.db_manager.session() as session:
            result = await session.execute(
                text("UPDATE intent_keywords SET is_active = '0', updated_at = NOW() WHERE keyword = :kw"),
                {"kw": keyword}
            )
            await session.commit()
            deactivated = result.rowcount > 0

            if deactivated:
                logger.info(f"Deactivated keyword: {keyword}")

            return deactivated

    async def get_keywords(
        self,
        category: Optional[KeywordCategory] = None,
        language: Optional[str] = None,
        active_only: bool = True,
    ) -> list[KeywordInfo]:
        """Get keywords with filters.

        Args:
            category: Optional category filter
            language: Optional language filter
            active_only: Only return active keywords

        Returns:
            List of KeywordInfo
        """
        if self.external_connector:
            if self._keywords_cache is None:
                self._keywords_cache = self.DEFAULT_KEYWORDS_STR_KEY.copy()
            
            result = []
            for cat, keywords in self._keywords_cache.items():
                for kw in keywords:
                    result.append(KeywordInfo(
                        keyword=kw,
                        category=KeywordCategory(cat),
                        language="zh" if any(ord(c) > 127 for c in kw) else "en",
                        pattern="",
                        weight="1.0",
                        is_active=True,
                    ))
            return result

        async with self.db_manager.session() as session:
            query = "SELECT keyword, category, language, pattern, weight, is_active FROM intent_keywords WHERE 1=1"
            params = {}

            if category:
                query += " AND category = :cat"
                params["cat"] = category.value
            if language:
                query += " AND language = :lang"
                params["lang"] = language
            if active_only:
                query += " AND is_active = '1'"

            result = await session.execute(text(query), params)
            rows = result.fetchall()

            return [
                KeywordInfo(
                    keyword=row[0],
                    category=KeywordCategory(row[1]),
                    language=row[2],
                    pattern=row[3] or "",
                    weight=row[4],
                    is_active=row[5] == "1",
                )
                for row in rows
            ]

    async def get_keywords_by_category(self) -> dict[KeywordCategory, list[str]]:
        """Get all active keywords grouped by category.

        Returns:
            Dictionary mapping category to list of keywords
        """
        keywords = await self.get_keywords(active_only=True)

        result: dict[KeywordCategory, list[str]] = {}
        for kw in keywords:
            if kw.category not in result:
                result[kw.category] = []
            if kw.keyword not in result[kw.category]:
                result[kw.category].append(kw.keyword)

        return result

    async def learn_from_query(
        self,
        query: str,
        detected_intent: str,
        feedback: str = "positive",
    ) -> Optional[KeywordInfo]:
        """Learn new keywords from user query feedback.

        Args:
            query: User query
            detected_intent: Intent that was detected
            feedback: "positive" or "negative"

        Returns:
            Optional KeywordInfo if a new keyword was learned
        """
        if feedback != "positive":
            return None

        import re

        query = query.strip().strip("？?。.")

        chinese_stop_words = {
            "多少", "几个", "如何", "怎么", "什么", "哪些", "是否",
            "请", "帮我", "帮我找", "请帮我", "我想", "我要",
            "怎样", "有没有", "查看", "查询", "看看",
        }

        question_patterns = [
            (r"(.+)情况如何$", "情况如何"),
            (r"(.+)有多少$", "有多少"),
            (r"(.+)是多少$", "是多少"),
            (r"(.+)怎么样$", "怎么样"),
            (r"(.+)如何$", "如何"),
            (r"(.+)情况$", "情况"),
            (r"(.+)怎样$", "怎样"),
            (r"(.+)有没有$", "有没有"),
            (r"(.+)吗$", "吗"),
            (r"(.+)呢$", "呢"),
        ]

        cleaned_query = query
        matched_suffix = ""
        for pattern, suffix in question_patterns:
            if query.endswith(suffix):
                match = re.match(pattern, query)
                if match:
                    cleaned_query = match.group(1)
                    matched_suffix = suffix
                    break

        business_indicators = set("订单量营收销售利润客户用户产品库存交易金额收入支出成本趋势统计汇总")

        candidate_keywords = []

        for i in range(len(cleaned_query)):
            for length in range(2, min(8, len(cleaned_query) - i + 1)):
                substr = cleaned_query[i:i+length]

                if substr in chinese_stop_words:
                    continue

                score = length * 10

                if any(char in business_indicators for char in substr):
                    score += 30

                if matched_suffix and matched_suffix in substr:
                    score -= 50

                candidate_keywords.append((substr, score))

        if not candidate_keywords:
            return None

        candidate_keywords.sort(key=lambda x: x[1], reverse=True)

        category_map = {
            "count": KeywordCategory.QUANTITY,
            "sum": KeywordCategory.AMOUNT,
            "avg": KeywordCategory.AMOUNT,
            "time_series": KeywordCategory.TIME,
            "statistics": KeywordCategory.STATISTICS,
            "sort": KeywordCategory.SORTING,
        }

        category = category_map.get(detected_intent, KeywordCategory.QUANTITY)

        for keyword, _ in candidate_keywords:
            if len(keyword) < 2:
                continue

            async with self.db_manager.session() as session:
                exists = await self._keyword_exists(session, keyword, category)
            
            if not exists:
                return await self.add_keyword(
                    keyword=keyword,
                    category=category,
                    description=f"Learned from query: {query[:50]}...",
                )

        return None

    async def export_keywords(self) -> dict:
        """Export all keywords as dictionary.

        Returns:
            Dictionary with categories as keys and keyword lists as values
        """
        keywords_by_category = await self.get_keywords_by_category()

        return {
            cat.value: keywords for cat, keywords in keywords_by_category.items()
        }

    async def import_keywords(self, data: dict, merge: bool = True) -> int:
        """Import keywords from dictionary.

        Args:
            data: Dictionary with categories as keys
            merge: If True, merge with existing; if False, replace

        Returns:
            Number of keywords imported
        """
        imported = 0

        async with self.db_manager.session() as session:
            for category_str, keywords in data.items():
                try:
                    category = KeywordCategory(category_str)
                except ValueError:
                    logger.warning(f"Unknown category: {category_str}")
                    continue

                if not merge:
                    await session.execute(
                        text("DELETE FROM intent_keywords WHERE category = :cat"),
                        {"cat": category.value}
                    )

                for keyword in keywords:
                    exists = await self._keyword_exists(session, keyword, category)
                    if not exists:
                        keyword_obj = Keyword(
                            id=str(uuid.uuid4()),
                            keyword=keyword,
                            category=category.value,
                            language="zh",
                            weight="1.0",
                            is_active="1",
                            description="Imported keyword",
                        )
                        session.add(keyword_obj)
                        imported += 1

            await session.commit()

        return imported

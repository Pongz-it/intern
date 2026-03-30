"""SQL Generator with safety checks.

Uses LLM to generate SQL queries from natural language,
with comprehensive security validation.
"""

import logging
import re
from dataclasses import dataclass
from typing import Optional, Any

from agent_rag.llm.interface import LLM

from agent_rag.text_to_sql.config import TextToSQLConfig
from agent_rag.text_to_sql.models import (
    DatabaseSchema,
    QueryIntent,
    SQLQuery,
    SQLQueryResult,
)

logger = logging.getLogger(__name__)


@dataclass
class SQLGenerationResult:
    """Result of SQL generation attempt."""

    success: bool
    sql: Optional[str] = None
    intent: Optional[QueryIntent] = None
    confidence: float = 0.0
    error_message: Optional[str] = None
    tables_used: list[str] = None
    columns_used: list[str] = None


class SQLGenerator:
    """Generates SQL queries from natural language using LLM.

    Features:
    - Schema-aware query generation
    - Comprehensive safety checks
    - SQL validation and repair
    - Hybrid column mapping (deterministic + LLM-assisted)
    """

    DANGEROUS_PATTERNS = [
        (r"\bDROP\b", "DROP statement"),
        (r"\bDELETE\b", "DELETE statement"),
        (r"\bUPDATE\b", "UPDATE statement"),
        (r"\bINSERT\b", "INSERT statement"),
        (r"\bALTER\b", "ALTER statement"),
        (r"\bCREATE\b", "CREATE statement"),
        (r"\bTRUNCATE\b", "TRUNCATE statement"),
        (r"\bGRANT\b", "GRANT statement"),
        (r"\bREVOKE\b", "REVOKE statement"),
        (r"\bEXECUTE\b", "EXECUTE statement"),
        (r"\bEXEC\b", "EXEC statement"),
        (r"\bUNION\b.*\bSELECT\b", "UNION-based injection"),
        (r"--.*$", "SQL comment injection"),
        (r"\/\*.*\*\/", "Block comment injection"),
        (r";\s*\w", "Multiple statements"),
    ]

    def __init__(
        self,
        llm: LLM,
        schema: DatabaseSchema,
        adapter: Any,
        config: Optional[TextToSQLConfig] = None,
        external_db_scanner: Optional[Any] = None,
    ):
        """Initialize the SQL generator.

        Args:
            llm: LLM instance for query generation
            schema: Database schema information
            adapter: Database adapter for validation
            config: Optional configuration
            external_db_scanner: Optional external database scanner for column mappings
        """
        self.llm = llm
        self.schema = schema
        self.adapter = adapter
        self.config = config or TextToSQLConfig.from_env()
        self.external_db_scanner = external_db_scanner

    def _build_column_mapping_section(self, table_name: str) -> str:
        """构建列名映射部分，使用混合方案。
        
        混合方案：
        1. 数据库扫描阶段：建立确定性映射（当前方案）
        2. SQL生成阶段：LLM结合映射生成正确的SQL
        
        Args:
            table_name: 表名
            
        Returns:
            str: 格式化的列名映射提示
        """
        mapping_lines = ["\n=== COLUMN MAPPING RULES (IMPORTANT!) ==="]
        
        if self.external_db_scanner:
            mapping_prompt = self.external_db_scanner.get_column_mapping_prompt(table_name)
            if mapping_prompt:
                mapping_lines.append(mapping_prompt)
                mapping_lines.append("")
        
        mapping_lines.extend([
            "COLUMN SELECTION RULES:",
            "1. You MUST use ONLY column names that EXACTLY match the columns listed in the schema",
            "2. Map user concepts to actual columns:",
            "   - 品牌/brand/厂商 → brand column",
            "   - 车型/car model/产品名称 → car_model column", 
            "   - 月份/month/时间 → month column",
            "   - 地区/region/区域 → region column",
            "   - 销量/sales/销售量 → sales_volume column",
            "   - 价格/price/售价 → price column",
            "3. If user mentions '产品名称' but schema has 'car_model', use 'car_model'",
            "4. If user mentions '日期' but schema has 'month', use 'month'",
            "5. If user mentions '品牌' but schema has 'brand', use 'brand'",
        ])
        
        mapping_lines.append("=== END COLUMN MAPPING RULES ===\n")
        
        return "\n".join(mapping_lines)

    def _build_table_schema_section(self, table_name: str) -> str:
        """构建表的Schema描述部分。
        
        Args:
            table_name: 表名
            
        Returns:
            str: 格式化的表结构描述
        """
        lines = []
        
        if self.external_db_scanner:
            table_desc = self.external_db_scanner.get_table_column_mapping_for_llm(table_name)
            if table_desc and table_desc != f"表 {table_name} 不存在":
                lines.append(table_desc)
                return "\n".join(lines)
        
        for table in self.schema.tables:
            if table.name != table_name:
                continue
                
            lines.append(f"TABLE: {table.name}")
            if table.description:
                lines.append(f"Description: {table.description}")
            lines.append("Columns:")
            
            for col in table.columns:
                col_desc = f"  - {col.name} ({col.column_type.value})"
                if col.is_primary_key:
                    col_desc += " [PRIMARY KEY]"
                if col.description:
                    col_desc += f" - {col.description}"
                lines.append(col_desc)
            
            break
        
        return "\n".join(lines) if lines else f"Table: {table_name}"

    def generate(
        self,
        natural_query: str,
        intent: Optional[QueryIntent] = None,
        suggested_tables: Optional[list[str]] = None,
        suggested_columns: Optional[list[str]] = None,
    ) -> SQLGenerationResult:
        """Generate SQL from natural language query.

        Args:
            natural_query: User's natural language query
            intent: Optional detected query intent
            suggested_tables: Optional list of suggested tables
            suggested_columns: Optional list of suggested columns

        Returns:
            SQLGenerationResult with generated SQL or error
        """
        try:
            schema_context = self._build_schema_context()
            system_prompt = self._build_system_prompt(intent, suggested_tables, suggested_columns)
            user_prompt = self._build_user_prompt(natural_query, schema_context)

            from agent_rag.llm.providers.litellm_provider import LLMMessage, ReasoningEffort

            messages = [
                LLMMessage(role="system", content=system_prompt),
                LLMMessage(role="user", content=user_prompt),
            ]

            response = self.llm.chat(
                messages,
                max_tokens=self.config.sql_generation_max_tokens,
                temperature=self.config.sql_generation_temperature,
                reasoning_effort=ReasoningEffort.OFF,
            )

            sql = self._extract_sql(response.content)

            if not sql:
                return SQLGenerationResult(
                    success=False,
                    error_message="Failed to extract SQL from response",
                )

            safety_result = self._safety_check(sql)
            if not safety_result["safe"]:
                return SQLGenerationResult(
                    success=False,
                    error_message=f"Safety check failed: {safety_result['reason']}",
                )

            if self.config.allowed_tables:
                tables_in_sql = self._extract_tables(sql)
                for table in tables_in_sql:
                    if table not in self.config.allowed_tables:
                        return SQLGenerationResult(
                            success=False,
                            error_message=f"Table {table} not in allowed list",
                        )

            validated_sql = self._validate_and_repair(sql)
            if validated_sql != sql:
                logger.info(f"SQL repaired: {sql} -> {validated_sql}")

            tables_used = self._extract_tables(validated_sql)
            columns_used = self._extract_columns(validated_sql)

            return SQLGenerationResult(
                success=True,
                sql=validated_sql,
                intent=intent or QueryIntent.FILTER,
                confidence=0.85,
                tables_used=tables_used,
                columns_used=columns_used,
            )

        except Exception as e:
            logger.error(f"SQL generation failed: {e}")
            return SQLGenerationResult(
                success=False,
                error_message=str(e),
            )

    def _build_schema_context(self) -> str:
        """Build schema context for the prompt."""
        context_parts = ["Available Tables:\n"]

        for table in self.schema.tables:
            cols = []
            for col in table.columns:
                col_desc = f"  - {col.name}: {col.column_type.value}"
                if col.is_primary_key:
                    col_desc += " (PRIMARY KEY)"
                if col.is_foreign_key:
                    col_desc += f" -> {col.foreign_table}.{col.foreign_column}"
                if col.description:
                    col_desc += f" ({col.description})"
                cols.append(col_desc)

            table_desc = f"Table: {table.name}"
            if table.description:
                table_desc += f" - {table.description}"

            context_parts.append(table_desc)
            context_parts.append("\n".join(cols))

            if table.relationships:
                context_parts.append("Relationships:")
                for rel in table.relationships:
                    context_parts.append(
                        f"  - {rel.from_table}.{rel.from_column} -> {rel.to_table}.{rel.to_column}"
                    )

            context_parts.append("")

        return "\n".join(context_parts)

    def _build_system_prompt(
        self,
        intent: Optional[QueryIntent] = None,
        suggested_tables: Optional[list[str]] = None,
        suggested_columns: Optional[list[str]] = None,
    ) -> str:
        """Build system prompt for SQL generation using hybrid mapping approach."""
        intent_hint = ""
        if intent:
            intent_hint = f"\nQuery Intent: {intent.value}"

        schema_overview = "=== DATABASE SCHEMA ===\n\n"

        for table in self.schema.tables:
            cols = []
            for col in table.columns:
                col_desc = f"  - {col.name} ({col.column_type.value})"
                if col.is_primary_key:
                    col_desc += " [PRIMARY KEY]"
                if col.description:
                    col_desc += f" - {col.description}"
                cols.append(col_desc)

            table_desc = f"TABLE: {table.name}\n"
            table_desc += f"Description: {table.description or 'Data table'}\n"
            table_desc += "Columns:\n" + "\n".join(cols)

            if table.relationships:
                table_desc += "\nRelationships:\n"
                for rel in table.relationships:
                    table_desc += f"  - {rel.from_table}.{rel.from_column} -> {rel.to_table}.{rel.to_column}\n"

            schema_overview += table_desc + "\n"

        suggested_tables_str = ", ".join(suggested_tables) if suggested_tables else "NONE - use your judgment based on the schema above"
        suggested_columns_str = ", ".join(suggested_columns) if suggested_columns else "NONE - use columns from the schema above"

        table_hint = f"""
TABLE AND COLUMN SELECTION RULES:
- Suggested tables (may or may not be relevant): {suggested_tables_str}
- Suggested columns (may or may not exist): {suggested_columns_str}

IMPORTANT COLUMN SELECTION GUIDELINES:
1. You MUST use ONLY column names that EXACTLY match the columns listed in the schema above
2. If the suggested columns do not exist in the schema, find the CLOSEST MATCH from available columns
3. For example, if query mentions "product_name" but schema has "car_model", use "car_model"
4. For example, if query mentions "date" but schema has "month", use "month"
5. When in doubt, prefer columns that directly match the user's intent

DECISION PROCESS:
1. First, review ALL available tables and columns in the schema above
2. Identify which tables contain information relevant to the user's query
3. From those tables, select columns that best match what the user is asking for
4. If user mentions a concept (like "product", "car", "vehicle", "date"), map it to the closest available column
5. If multiple tables could work, prefer the table that most directly answers the question"""

        return f"""{schema_overview}{table_hint}

You are an expert SQL generator. Your task is to convert natural language queries into accurate, safe PostgreSQL queries.

TASK:
1. Analyze the user's query to understand what information they want
2. Review the schema above to find relevant tables and columns
3. Select ONLY columns that exist in the schema (do NOT invent column names)
4. Generate a PostgreSQL query that retrieves the information needed to answer the user's question

CRITICAL RULES:
1.优先使用 SQL 查询 (use SQL queries FIRST) to retrieve structured data
2. You MUST use ONLY table and column names that EXACTLY match those in the schema
3. If a concept mentioned by the user doesn't exist in the schema, find the closest matching column name
4. Generate ONLY SELECT statements - no INSERT, UPDATE, DELETE, DROP, or other operations
5. Always include a LIMIT clause (default: 100)
6. Use appropriate aggregate functions (COUNT, SUM, AVG, MIN, MAX) for statistical queries
7. Use proper PostgreSQL syntax with correct table/column names

{self._build_column_mapping_section(suggested_tables[0]) if suggested_tables else ""}

COMMON COLUMN MAPPINGS (use these when user mentions similar concepts):
- For vehicle/car/mobile product names → use "car_model" or "model" columns
- For time periods (date/time/when) → use "month", "created_at", or "order_date" columns  
- For quantities/volume → use "sales_volume", "quantity", or "amount" columns
- For customer/client user info → use columns from "customers" or "test_customers" table
- For products/goods items → use "name" or "product_name" from "products" table
- For regions/areas/locations → use "region" column
- For brand/company/car make names → use "brand" column in vehicle_sales_2024 (contains values like "理想", "比亚迪", "特斯拉", "蔚来", "小鹏", "大众")

FILTERING EXAMPLES (IMPORTANT!):
- "理想销量" or "理想汽车销量" → SELECT SUM(sales_volume) FROM vehicle_sales_2024 WHERE brand = '理想'
- "比亚迪秦的销量" → SELECT SUM(sales_volume) FROM vehicle_sales_2024 WHERE brand = '比亚迪' AND car_model LIKE '%秦%'
- "特斯拉Model Y的销量" → SELECT SUM(sales_volume) FROM vehicle_sales_2024 WHERE brand = '特斯拉' AND car_model = 'Model Y'
- "华东地区销量" → SELECT SUM(sales_volume) FROM vehicle_sales_2024 WHERE region = '华东'
- "3月销量" → SELECT SUM(sales_volume) FROM vehicle_sales_2024 WHERE month = 3

If the user's query can be answered using the available tables and columns, generate a SQL query.
If the requested information does NOT exist in any of the available tables, respond with {{"sql": null, "reason": "cannot_answer"}}.

Example mappings:
- "car sales" → vehicle_sales_2024.sales_volume, vehicle_sales_2024.car_model
- "product name" → products.name (NOT "product_name" unless it exists)
- "order date" → orders.order_date (NOT "date")
- "customer spending" → test_customers.consumption_amount

{intent_hint}

Respond ONLY with a JSON object:
{{"sql": "SELECT ..."}}"""

    def _build_user_prompt(
        self, natural_query: str, schema_context: str
    ) -> str:
        """Build user prompt for SQL generation."""
        return f"""{schema_context}

User Query: {natural_query}

Generate a SQL query to answer this question. Respond ONLY with a JSON object:
{{"sql": "SELECT ..."}}"""

    def _extract_sql(self, response: str) -> Optional[str]:
        """Extract SQL from LLM response."""
        if not response:
            return None

        content = response.strip()

        if content.startswith("```sql"):
            content = content[6:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]

        content = content.strip()

        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            import json

            try:
                obj = json.loads(json_match.group())
                if isinstance(obj, dict) and "sql" in obj:
                    sql = obj["sql"]
                    if sql and sql.lower() != "null":
                        return sql
            except json.JSONDecodeError:
                pass

        if content.upper().startswith("SELECT"):
            return content

        return None

    def _safety_check(self, sql: str) -> dict:
        """Perform safety check on SQL."""
        sql_upper = sql.upper()

        for pattern, description in self.DANGEROUS_PATTERNS:
            if re.search(pattern, sql_upper, re.IGNORECASE):
                return {"safe": False, "reason": description}

        if "SELECT" not in sql_upper:
            return {"safe": False, "reason": "No SELECT statement"}

        if re.search(r"LIMIT\s*,\s*\d", sql_upper, re.IGNORECASE):
            return {"safe": False, "reason": "Invalid LIMIT clause"}

        if re.search(r"\bLIMIT\s+0\b", sql_upper, re.IGNORECASE):
            return {"safe": False, "reason": "LIMIT 0 not allowed"}

        return {"safe": True, "reason": None}

    def _extract_tables(self, sql: str) -> list[str]:
        """Extract table names from SQL."""
        tables = set()

        from_pattern = r"\bFROM\s+([a-zA-Z_][a-zA-Z0-9_]*)"
        join_pattern = r"\bJOIN\s+([a-zA-Z_][a-zA-Z0-9_]*)"

        for match in re.finditer(from_pattern, sql, re.IGNORECASE):
            tables.add(match.group(1))

        for match in re.finditer(join_pattern, sql, re.IGNORECASE):
            tables.add(match.group(1))

        return list(tables)

    def _extract_columns(self, sql: str) -> list[str]:
        """Extract column names from SQL."""
        columns = set()

        select_pattern = r"SELECT\s+(.+?)\s+FROM"
        select_match = re.search(select_pattern, sql, re.IGNORECASE | re.DOTALL)
        if select_match:
            select_part = select_match.group(1)
            col_pattern = r"([a-zA-Z_][a-zA-Z0-9_]*)\."
            for match in re.finditer(col_pattern, select_part):
                columns.add(match.group(1))

        return list(columns)

    def _validate_and_repair(self, sql: str) -> str:
        """Validate and repair SQL syntax."""
        repaired = sql

        if "LIMIT" not in sql.upper():
            repaired = f"{sql} LIMIT {self.config.default_limit}"

        limit_match = re.search(r"LIMIT\s+(\d+)", sql, re.IGNORECASE)
        if limit_match:
            limit_value = int(limit_match.group(1))
            if limit_value > self.config.max_query_results:
                repaired = re.sub(
                    r"LIMIT\s+\d+",
                    f"LIMIT {self.config.max_query_results}",
                    repaired,
                    flags=re.IGNORECASE,
                )

        return repaired


class SQLQueryEngine:
    """Complete SQL query engine with generation and execution."""

    def __init__(
        self,
        llm: LLM,
        schema: DatabaseSchema,
        adapter: Any,
        config: Optional[TextToSQLConfig] = None,
        external_db_scanner: Optional[Any] = None,
    ):
        """Initialize the query engine.
        
        Args:
            llm: LLM instance for query generation
            schema: Database schema information
            adapter: Database adapter for execution
            config: Optional configuration
            external_db_scanner: Optional external database scanner for column mappings
        """
        self.generator = SQLGenerator(
            llm, 
            schema, 
            adapter, 
            config,
            external_db_scanner=external_db_scanner,
        )
        self.adapter = adapter
        self.schema = schema

    async def execute(
        self,
        natural_query: str,
        intent: Optional[QueryIntent] = None,
        suggested_tables: Optional[list[str]] = None,
        suggested_columns: Optional[list[str]] = None,
    ) -> SQLQueryResult:
        """Execute a natural language query.

        Args:
            natural_query: User's natural language query
            intent: Optional detected query intent
            suggested_tables: Optional list of suggested tables
            suggested_columns: Optional list of suggested columns

        Returns:
            SQLQueryResult with query results
        """
        generation_result = self.generator.generate(
            natural_query, intent, suggested_tables, suggested_columns
        )

        if not generation_result.success:
            return SQLQueryResult(
                sql="",
                success=False,
                error_message=generation_result.error_message,
            )

        result = await self.adapter.execute_query(generation_result.sql)

        return SQLQueryResult(
            sql=generation_result.sql,
            success=result.success,
            row_count=result.row_count,
            data=result.data,
            error_message=result.error_message,
            execution_time_ms=result.execution_time_ms,
        )

    def get_schema(self) -> DatabaseSchema:
        """Get the database schema."""
        return self.schema

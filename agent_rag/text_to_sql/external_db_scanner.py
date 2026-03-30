"""外部数据库扫描器。

自动扫描外部数据库，获取表名、列名、索引等元数据信息，
用于关键词注入和混合查询触发。
支持多种数据库类型：PostgreSQL, MySQL, SQLite, Oracle, SQL Server, Snowflake, BigQuery, DuckDB
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from agent_rag.core.database import DatabaseManager, get_db_manager
from agent_rag.core.external_database_connector import ExternalDatabaseConnector
from agent_rag.text_to_sql.models import (
    ColumnType,
    DatabaseSchema,
    DatabaseTable,
    TableColumn,
)
from agent_rag.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DiscoveredTable:
    """已发现的表信息"""
    name: str
    description: str = ""
    columns: list["DiscoveredColumn"] = field(default_factory=list)
    row_count: Optional[int] = None
    sample_data: list[dict] = field(default_factory=list)
    is_external: bool = False
    source_database: str = ""


@dataclass
class DiscoveredColumn:
    """已发现的列信息"""
    name: str
    column_type: str
    description: str = ""
    is_primary_key: bool = False
    is_foreign_key: bool = False
    is_nullable: bool = True
    default_value: Optional[str] = None
    sample_values: list[str] = field(default_factory=list)
    synonyms: list[str] = field(default_factory=list)


@dataclass
class DiscoveredDatabase:
    """已发现的数据库信息"""
    name: str
    tables: list[DiscoveredTable] = field(default_factory=list)
    connection_info: dict = field(default_factory=dict)
    last_scanned: Optional[str] = None


class ExternalDatabaseScanner:
    """外部数据库扫描器。
    
    支持扫描外部数据库的表结构、列信息、示例数据等，
    并提供关键词提取功能用于意图分析和SQL生成。
    """
    
    TYPE_SYNONYMS = {
        "varchar": ["字符串", "文本", "文本字段", "string", "text"],
        "text": ["长文本", "描述", "内容", "详情"],
        "integer": ["整数", "数字", "int", "numeric"],
        "int": ["整数", "数字"],
        "bigint": ["大整数", "长数字"],
        "smallint": ["小整数"],
        "decimal": ["小数", "精确数字", "金额"],
        "numeric": ["数字", "数值"],
        "float": ["浮点数", "小数"],
        "double": ["双精度浮点数"],
        "boolean": ["布尔", "是/否", "开关", "bool"],
        "bool": ["布尔", "是否"],
        "date": ["日期", "几号"],
        "time": ["时间", "几点"],
        "timestamp": ["时间戳", "日期时间", "datetime"],
        "timestamptz": ["带时区时间", "UTC时间"],
        "json": ["JSON数据", "对象数据", "结构化数据"],
        "jsonb": ["二进制JSON", "压缩JSON"],
        "uuid": ["唯一标识", "ID类型"],
        "array": ["数组", "列表"],
    }
    
    COMMON_FIELD_SYNONYMS = {
        "id": ["编号", "序号", "序列号", "唯一标识", "标识符", "主键"],
        "name": ["名称", "名字", "姓名", "标题", "名称字段"],
        "title": ["标题", "主题", "名称", "名字"],
        "description": ["描述", "说明", "详情", "备注", "内容"],
        "amount": ["金额", "数量", "总数", "费用", "价格"],
        "price": ["价格", "金额", "费用", "售价"],
        "quantity": ["数量", "数目", "个数", "件数"],
        "status": ["状态", "情况", "状况"],
        "type": ["类型", "种类", "类别"],
        "category": ["分类", "类别", "种类"],
        "created_at": ["创建时间", "创建日期", "添加时间", "何时创建"],
        "updated_at": ["更新时间", "修改时间", "最后更新"],
        "deleted_at": ["删除时间", "删除日期"],
        "is_active": ["是否启用", "激活状态", "启用状态"],
        "is_deleted": ["是否删除", "删除标记"],
        "user_id": ["用户ID", "用户编号", "用户标识"],
        "customer_id": ["客户ID", "客户编号", "顾客ID"],
        "order_id": ["订单ID", "订单编号", "订单号"],
        "product_id": ["产品ID", "商品ID", "产品编号"],
        "email": ["邮箱", "电子邮件", "邮件地址"],
        "phone": ["电话", "手机", "手机号", "联系电话"],
        "address": ["地址", "住址", "位置"],
        "consumption_amount": ["消费金额", "消费", "花费", "支出", "金额"],
        "is_vip": ["VIP", "会员", "会员等级", "等级"],
        "sales": ["销量", "销售量", "销售额", "销售"],
        "sales_volume": ["销量", "销售量", "销售总额", "销售数量", "销售量"],
        "revenue": ["收入", "营收", "销售额", "营业额"],
    }
    
    VEHICLE_FIELD_SYNONYMS = {
        "brand": ["品牌", "汽车品牌", "车品牌", "厂商", "制造商", "公司", "厂家"],
        "car_model": ["车型", "车型名称", "汽车型号", "车辆型号", "车系", "产品名称", "产品型号", "型号", "车款"],
        "model": ["车型", "型号", "产品型号"],
        "vehicle_model": ["车型", "车辆型号"],
        "month": ["月份", "月", "时间", "日期", "周期"],
        "year": ["年份", "年", "年度"],
        "region": ["区域", "地区", "地域", "省份", "城市", "市场", "区域划分"],
        "area": ["区域", "地区", "面积"],
        "sales_volume": ["销量", "销售量", "销售数量", "销售额", "销售总额", "卖出数量", "交易量"],
        "price": ["价格", "售价", "单价", "成交价"],
        "total_sales": ["总销量", "总销售量", "累计销量", "总计销量"],
        "total_amount": ["总金额", "总额", "总计"],
        "created_at": ["创建时间", "录入时间", "记录时间", "数据时间"],
    }
    
    VEHICLE_BRAND_SYNONYMS = {
        "比亚迪": ["BYD", "byd", "比亚迪汽车"],
        "特斯拉": ["Tesla", "tesla", "特斯拉汽车", "Tesla汽车"],
        "理想": ["理想汽车", "Li", "li-auto", "Li Auto", "L系列"],
        "蔚来": ["NIO", "nio", "蔚来汽车"],
        "小鹏": ["XPeng", "xpeng", "小鹏汽车", "小鹏P系列"],
        "大众": ["Volkswagen", "VW", "vw", "大众汽车"],
        "丰田": ["Toyota", "toyota", "丰田汽车"],
        "本田": ["Honda", "honda", "本田汽车"],
        "福特": ["Ford", "ford", "福特汽车"],
        "奔驰": ["Mercedes", "Mercedes-Benz", "奔驰汽车"],
        "宝马": ["BMW", "bmw", "宝马汽车"],
    }
    
    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        external_connector: Optional[Any] = None,
        cache_dir: str = "./data/db_discoveries",
    ):
        """初始化数据库扫描器。
        
        Args:
            db_manager: 数据库管理器（用于内置数据库）
            external_connector: 外部数据库连接器
            cache_dir: 扫描结果缓存目录
        """
        self.db_manager = db_manager
        self.external_connector = external_connector
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self._discovered_databases: dict[str, DiscoveredDatabase] = {}
        self._table_column_mapping: dict[str, list[str]] = {}
        self._all_keywords: set[str] = set()
        self._column_synonyms: dict[str, list[str]] = {}
        self._table_synonyms: dict[str, list[str]] = {}
        
        if external_connector:
            self._db_type = getattr(external_connector, 'db_type', 'postgresql')
        else:
            self._db_type = "postgresql"
    
    @property
    def db_type(self) -> str:
        """获取数据库类型。"""
        if self.external_connector:
            return getattr(self.external_connector, 'db_type', 'postgresql')
        return "postgresql"
    
    async def scan_database(
        self,
        db_name: str,
        db_manager: Optional[DatabaseManager] = None,
        include_sample_data: bool = True,
        sample_size: int = 5,
    ) -> DiscoveredDatabase:
        """扫描指定数据库的所有表和列。
        
        Args:
            db_name: 数据库名称标识
            db_manager: 数据库管理器实例（可选）
            include_sample_data: 是否包含示例数据
            sample_size: 示例数据条数
            
        Returns:
            DiscoveredDatabase: 包含所有表信息的数据库对象
        """
        logger.info(f"[DBScanner] Scanning database: {db_name}")
        
        discovered_db = DiscoveredDatabase(name=db_name)
        
        try:
            if db_manager:
                tables = await self._scan_internal_db(db_manager, include_sample_data, sample_size)
            elif self.external_connector:
                tables = await self._scan_external_db(include_sample_data, sample_size)
            else:
                logger.warning("[DBScanner] No database manager or connector provided")
                return discovered_db
            
            discovered_db.tables = tables
            discovered_db.last_scanned = self._get_timestamp()
            
            self._discovered_databases[db_name] = discovered_db
            self._build_keyword_index(tables, db_name)
            
            logger.info(f"[DBScanner] Discovered {len(tables)} tables in {db_name}")
            
        except Exception as e:
            logger.error(f"[DBScanner] Failed to scan database {db_name}: {e}")
        
        return discovered_db
    
    async def _scan_internal_db(
        self,
        db_manager: DatabaseManager,
        include_sample_data: bool,
        sample_size: int,
    ) -> list[DiscoveredTable]:
        """扫描内置数据库。"""
        tables = []
        
        try:
            async with db_manager.session() as session:
                result = await session.execute(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = 'public'"
                )
                table_names = [row[0] for row in result.fetchall()]
                
                for table_name in table_names:
                    table = await self._extract_table_info(
                        table_name, db_manager, 
                        include_sample_data, sample_size,
                        is_external=False
                    )
                    tables.append(table)
                    
        except Exception as e:
            logger.error(f"[DBScanner] Failed to scan internal database: {e}")
        
        return tables
    
    async def _scan_external_db(
        self,
        include_sample_data: bool,
        sample_size: int,
    ) -> list[DiscoveredTable]:
        """扫描外部数据库。"""
        tables = []
        
        if not self.external_connector:
            return tables
        
        try:
            from sqlalchemy import text
            
            with self.external_connector.session() as session:
                result = session.execute(text(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = 'public'"
                ))
                table_names = [row[0] for row in result.fetchall()]
                
                for table_name in table_names:
                    table = await self._extract_table_info_external(
                        table_name, session,
                        include_sample_data, sample_size,
                        is_external=True
                    )
                    tables.append(table)
                    
        except Exception as e:
            logger.error(f"[DBScanner] Failed to scan external database: {e}")
        
        return tables
    
    async def _extract_table_info(
        self,
        table_name: str,
        db_manager: DatabaseManager,
        include_sample_data: bool,
        sample_size: int,
        is_external: bool,
    ) -> DiscoveredTable:
        """提取单个表的详细信息。"""
        table = DiscoveredTable(
            name=table_name,
            is_external=is_external,
            source_database="internal" if not is_external else "external"
        )
        
        try:
            async with db_manager.session() as session:
                result = await session.execute(
                    f"SELECT column_name, data_type, column_default, is_nullable, "
                    f"ordinal_position FROM information_schema.columns "
                    f"WHERE table_name = '{table_name}' ORDER BY ordinal_position"
                )
                columns = result.fetchall()
                
                for col in columns:
                    col_name, data_type, default_val, nullable, position = col
                    column = DiscoveredColumn(
                        name=col_name,
                        column_type=data_type,
                        is_nullable=(nullable == "YES"),
                        default_value=default_val,
                    )
                    
                    self._column_synonyms[col_name] = self._build_column_synonyms(
                        col_name, data_type
                    )
                    table.columns.append(column)
                
                if include_sample_data:
                    sample_data = await self._fetch_sample_data(
                        db_manager, table_name, sample_size
                    )
                    table.sample_data = sample_data
                    
                    for col in table.columns:
                        col.sample_values = [
                            str(row.get(col.name, "")) 
                            for row in sample_data[:3]
                            if col.name in row
                        ]
                    
                table.description = self._generate_table_description(table)
                
        except Exception as e:
            logger.warning(f"[DBScanner] Failed to extract table {table_name}: {e}")
        
        return table
    
    async def _extract_table_info_external(
        self,
        table_name: str,
        session,
        include_sample_data: bool,
        sample_size: int,
        is_external: bool,
    ) -> DiscoveredTable:
        """提取外部数据库单个表的详细信息。"""
        from sqlalchemy import text
        
        table = DiscoveredTable(
            name=table_name,
            is_external=is_external,
            source_database="external"
        )
        
        try:
            result = session.execute(text(
                "SELECT column_name, data_type, column_default, is_nullable, "
                "ordinal_position FROM information_schema.columns "
                "WHERE table_name = :table_name ORDER BY ordinal_position"
            ), {"table_name": table_name})
            columns = result.fetchall()
            
            for col in columns:
                col_name, data_type, default_val, nullable, position = col
                column = DiscoveredColumn(
                    name=col_name,
                    column_type=data_type,
                    is_nullable=(nullable == "YES"),
                    default_value=default_val,
                )
                
                self._column_synonyms[col_name] = self._build_column_synonyms(
                    col_name, data_type
                )
                table.columns.append(column)
            
            if include_sample_data:
                sample_data = self._fetch_sample_data_external(
                    session, table_name, sample_size
                )
                table.sample_data = sample_data
                
                for col in table.columns:
                    col.sample_values = [
                        str(row.get(col.name, "")) 
                        for row in sample_data[:3]
                        if col.name in row
                    ]
                
            table.description = self._generate_table_description(table)
            
        except Exception as e:
            logger.warning(f"[DBScanner] Failed to extract table {table_name}: {e}")
        
        return table
    
    def _build_column_synonyms(self, column_name: str, data_type: str) -> list[str]:
        """构建列名的同义词列表。"""
        synonyms = []
        col_lower = column_name.lower()
        
        if col_lower in self.COMMON_FIELD_SYNONYMS:
            synonyms.extend(self.COMMON_FIELD_SYNONYMS[col_lower])
        
        if col_lower in self.VEHICLE_FIELD_SYNONYMS:
            synonyms.extend(self.VEHICLE_FIELD_SYNONYMS[col_lower])
        
        if data_type.lower() in self.TYPE_SYNONYMS:
            synonyms.extend(self.TYPE_SYNONYMS[data_type.lower()])
        
        return list(set(synonyms))
    
    def _build_table_synonyms(self, table_name: str) -> list[str]:
        """构建表名的同义词。"""
        synonyms = []
        
        table_lower = table_name.lower()
        
        if "customer" in table_lower:
            synonyms.extend(["客户", "顾客", "用户", "会员"])
        if "order" in table_lower:
            synonyms.extend(["订单", "订购", "订货"])
        if "product" in table_lower:
            synonyms.extend(["产品", "商品", "物品", "货品"])
        if "user" in table_lower:
            synonyms.extend(["用户", "使用者", "账号"])
        if "transaction" in table_lower:
            synonyms.extend(["交易", "转账", "收支"])
        if "payment" in table_lower:
            synonyms.extend(["支付", "付款", "结算"])
        if "inventory" in table_lower:
            synonyms.extend(["库存", "存货"])
        if "report" in table_lower:
            synonyms.extend(["报告", "报表", "统计"])
        if "log" in table_lower:
            synonyms.extend(["日志", "记录"])
        if "vehicle" in table_lower:
            synonyms.extend(["汽车", "车辆", "车型", "车"])
        if "sales" in table_lower:
            synonyms.extend(["销量", "销售", "营业额", "销售额"])
        if "car" in table_lower:
            synonyms.extend(["汽车", "车辆", "车型", "车"])
        
        return synonyms
    
    async def _fetch_sample_data(
        self,
        db_manager: DatabaseManager,
        table_name: str,
        limit: int,
    ) -> list[dict]:
        """获取表示例数据。"""
        try:
            async with db_manager.session() as session:
                result = await session.execute(
                    f"SELECT * FROM {table_name} LIMIT {limit}"
                )
                rows = result.fetchall()
                columns = result.keys()
                
                return [
                    dict(zip(columns, row))
                    for row in rows
                ]
        except Exception:
            return []
    
    def _fetch_sample_data_external(
        self,
        session,
        table_name: str,
        limit: int,
    ) -> list[dict]:
        """获取外部数据库表示例数据。"""
        from sqlalchemy import text
        
        try:
            result = session.execute(
                text(f"SELECT * FROM {table_name} LIMIT :limit"),
                {"limit": limit}
            )
            rows = result.fetchall()
            columns = result.keys()
            
            return [
                dict(zip(columns, row))
                for row in rows
            ]
        except Exception:
            return []
    
    def _generate_table_description(self, table: DiscoveredTable) -> str:
        """生成表的自然语言描述。"""
        col_names = [col.name for col in table.columns[:5]]
        return f"Table {table.name} with columns: {', '.join(col_names)}"
    
    def get_column_mapping_hints(self, table_name: str) -> dict[str, list[str]]:
        """获取列名映射提示，用于SQL生成时的列名映射。
        
        Returns:
            dict: {概念: [可能的列名列表]}
        """
        mapping_hints = {}
        
        if table_name not in self._table_column_mapping:
            return mapping_hints
        
        columns = self._table_column_mapping[table_name]
        
        concept_to_columns = {
            "品牌": ["brand", "car_brand", "manufacturer", "make"],
            "车型": ["car_model", "model", "vehicle_model", "product_name", "name"],
            "月份": ["month", "order_month", "sale_month", "date"],
            "时间": ["created_at", "order_date", "sale_date", "date", "time"],
            "地区": ["region", "area", "province", "city", "location"],
            "销量": ["sales_volume", "sales", "quantity", "amount", "total_sales"],
            "价格": ["price", "unit_price", "amount", "cost"],
            "金额": ["amount", "total_amount", "price", "sales_amount", "consumption_amount"],
            "客户": ["customer_id", "user_id", "customer_name", "user_name"],
            "产品": ["product_id", "product_name", "name", "item_name"],
            "订单": ["order_id", "order_no", "order_number"],
            "状态": ["status", "order_status", "payment_status"],
            "日期": ["created_at", "order_date", "sale_date", "date"],
            "总数": ["total", "total_amount", "total_quantity", "sum"],
            "平均值": ["avg", "average", "avg_amount"],
        }
        
        for col in columns:
            col_lower = col.lower()
            for concept, possible_cols in concept_to_columns.items():
                if col_lower in possible_cols:
                    if concept not in mapping_hints:
                        mapping_hints[concept] = []
                    if col not in mapping_hints[concept]:
                        mapping_hints[concept].append(col)
        
        return mapping_hints
    
    def get_all_column_mappings(self) -> dict[str, dict[str, list[str]]]:
        """获取所有表的列名映射提示。
        
        Returns:
            dict: {表名: {概念: [列名列表]}}
        """
        all_mappings = {}
        for table_name in self._table_column_mapping.keys():
            all_mappings[table_name] = self.get_column_mapping_hints(table_name)
        return all_mappings
    
    def get_column_mapping_prompt(self, table_name: str) -> str:
        """生成列名映射提示文本，用于LLM SQL生成。
        
        Args:
            table_name: 表名
            
        Returns:
            str: 格式化的映射提示文本
        """
        mapping = self.get_column_mapping_hints(table_name)
        
        if not mapping:
            return ""
        
        prompt_lines = ["列名映射参考:"]
        
        concept_order = ["品牌", "车型", "月份", "时间", "地区", "销量", "价格", "金额", "客户", "产品", "订单", "状态", "日期", "总数", "平均值"]
        
        for concept in concept_order:
            if concept in mapping:
                columns = mapping[concept]
                prompt_lines.append(f"  - {concept} → {', '.join(columns)}")
        
        return "\n".join(prompt_lines)
    
    def get_table_column_mapping_for_llm(self, table_name: str) -> str:
        """生成表和列的完整描述，用于LLM理解数据库结构。
        
        Args:
            table_name: 表名
            
        Returns:
            str: 格式化的表结构描述
        """
        if table_name not in self._table_column_mapping:
            return f"表 {table_name} 不存在"
        
        columns = self._table_column_mapping[table_name]
        
        lines = [f"表: {table_name}", "列:"]
        
        for col in columns:
            col_lower = col.lower()
            synonyms = self._column_synonyms.get(col, [])
            synonym_str = f" (同义词: {', '.join(synonyms)})" if synonyms else ""
            lines.append(f"  - {col}{synonym_str}")
        
        mapping_prompt = self.get_column_mapping_prompt(table_name)
        if mapping_prompt:
            lines.append("")
            lines.append(mapping_prompt)
        
        return "\n".join(lines)
    
    def _build_keyword_index(
        self,
        tables: list[DiscoveredTable],
        db_name: str,
    ):
        """构建关键词索引。"""
        for table in tables:
            self._all_keywords.add(table.name)
            self._all_keywords.add(table.name.lower())
            
            self._table_synonyms[table.name] = self._build_table_synonyms(
                table.name
            )
            self._all_keywords.update(self._table_synonyms[table.name])
            
            self._table_column_mapping[table.name] = [
                col.name for col in table.columns
            ]
            
            for col in table.columns:
                self._all_keywords.add(col.name)
                self._all_keywords.add(col.name.lower())
                
                self._all_keywords.update(self._column_synonyms.get(col.name, []))
    
    def get_all_keywords(self) -> set[str]:
        """获取所有发现的关键字。"""
        return self._all_keywords.copy()
    
    def get_table_column_mapping(self) -> dict[str, list[str]]:
        """获取表到列的映射。"""
        return self._table_column_mapping.copy()
    
    def get_column_synonyms(self, column_name: str) -> list[str]:
        """获取列名的同义词。"""
        return self._column_synonyms.get(column_name, [])
    
    def get_table_synonyms(self, table_name: str) -> list[str]:
        """获取表名的同义词。"""
        return self._table_synonyms.get(table_name, [])
    
    def detect_database_keywords(self, query: str) -> dict:
        """检测查询中包含的数据库关键字。
        
        Args:
            query: 用户查询文本
            
        Returns:
            包含检测到的表名、列名、同义词匹配等信息
        """
        query_lower = query.lower()
        detected_tables = []
        detected_columns = []
        matched_synonyms = []
        
        for table_name in self._table_column_mapping.keys():
            table_lower = table_name.lower()
            
            if table_lower in query_lower:
                detected_tables.append(table_name)
                matched_synonyms.extend(self._table_synonyms.get(table_name, []))
            
            for synonym in self._table_synonyms.get(table_name, []):
                if synonym.lower() in query_lower:
                    detected_tables.append(table_name)
                    matched_synonyms.append(synonym)
                    break
        
        for column_name in self._column_synonyms.keys():
            column_lower = column_name.lower()
            
            if column_lower in query_lower:
                detected_columns.append(column_name)
                matched_synonyms.extend(self._column_synonyms.get(column_name, []))
            
            for synonym in self._column_synonyms.get(column_name, []):
                if synonym.lower() in query_lower:
                    if column_name not in detected_columns:
                        detected_columns.append(column_name)
                    matched_synonyms.append(synonym)
        
        vehicle_brand_columns = ["brand", "car_brand", "vehicle_brand"]
        for brand_chinese, brand_synonyms in self.VEHICLE_BRAND_SYNONYMS.items():
            for synonym in [brand_chinese] + brand_synonyms:
                if synonym.lower() in query_lower:
                    matched_synonyms.append(f"品牌:{brand_chinese}")
                    for col in vehicle_brand_columns:
                        if col in self._column_synonyms:
                            if col not in detected_columns:
                                detected_columns.append(col)
                    break
        
        for keyword in self._all_keywords:
            if keyword.lower() in query_lower:
                if keyword in self._table_column_mapping:
                    if keyword not in detected_tables:
                        detected_tables.append(keyword)
                if keyword not in detected_columns:
                    detected_columns.append(keyword)
        
        return {
            "detected_tables": list(set(detected_tables)),
            "detected_columns": list(set(detected_columns)),
            "matched_synonyms": list(set(matched_synonyms)),
            "has_database_reference": (
                len(detected_tables) > 0 or 
                len(detected_columns) > 0 or 
                len(matched_synonyms) > 0
            ),
        }
    
    def save_discovery(self, db_name: str) -> str:
        """保存扫描结果到缓存文件。"""
        if db_name not in self._discovered_databases:
            return ""
        
        cache_file = self.cache_dir / f"{db_name}.json"
        
        db = self._discovered_databases[db_name]
        data = {
            "name": db.name,
            "tables": [
                {
                    "name": t.name,
                    "description": t.description,
                    "columns": [
                        {
                            "name": c.name,
                            "type": c.column_type,
                            "is_nullable": c.is_nullable,
                            "sample_values": c.sample_values,
                        }
                        for c in t.columns
                    ],
                    "sample_data": t.sample_data[:3],
                    "is_external": t.is_external,
                }
                for t in db.tables
            ],
            "last_scanned": db.last_scanned,
        }
        
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"[DBScanner] Saved discovery to {cache_file}")
        return str(cache_file)
    
    def load_discovery(self, db_name: str) -> Optional[DiscoveredDatabase]:
        """从缓存文件加载扫描结果。"""
        cache_file = self.cache_dir / f"{db_name}.json"
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            db = DiscoveredDatabase(name=data["name"])
            db.last_scanned = data.get("last_scanned")
            
            for t_data in data.get("tables", []):
                table = DiscoveredTable(
                    name=t_data["name"],
                    description=t_data.get("description", ""),
                    is_external=t_data.get("is_external", False),
                )
                
                for c_data in t_data.get("columns", []):
                    column = DiscoveredColumn(
                        name=c_data["name"],
                        column_type=c_data["type"],
                        is_nullable=c_data.get("is_nullable", True),
                        sample_values=c_data.get("sample_values", []),
                    )
                    table.columns.append(column)
                
                table.sample_data = t_data.get("sample_data", [])
                db.tables.append(table)
            
            self._discovered_databases[db_name] = db
            self._build_keyword_index(db.tables, db_name)
            
            logger.info(f"[DBScanner] Loaded discovery from {cache_file}")
            return db
            
        except Exception as e:
            logger.error(f"[DBScanner] Failed to load discovery: {e}")
            return None
    
    def _get_timestamp(self) -> str:
        """获取当前时间戳。"""
        from datetime import datetime
        return datetime.now().isoformat()


class DBKeywordInjector:
    """数据库关键词注入器。
    
    将发现的数据库表名、列名及其同义词注入到
    意图分析和SQL生成流程中，确保当用户提到
    相关关键词时能正确触发数据库查询。
    """
    
    def __init__(self, scanner: ExternalDatabaseScanner):
        """初始化关键词注入器。
        
        Args:
            scanner: 数据库扫描器实例
        """
        self.scanner = scanner
        self._injected_keywords: set[str] = set()
    
    def inject_keywords(
        self,
        query: str,
    ) -> tuple[str, dict]:
        """注入数据库关键词到查询中。
        
        Args:
            query: 原始用户查询
            
        Returns:
            tuple: (增强后的查询, 注入元数据)
        """
        enhanced_query = query
        injection_metadata = {
            "original_query": query,
            "detected_tables": [],
            "detected_columns": [],
            "added_context": [],
        }
        
        detection_result = self.scanner.detect_database_keywords(query)
        
        if detection_result["has_database_reference"]:
            injection_metadata["detected_tables"] = detection_result["detected_tables"]
            injection_metadata["detected_columns"] = detection_result["detected_columns"]
            
            for table in detection_result["detected_tables"]:
                if table not in self._injected_keywords:
                    self._injected_keywords.add(table)
                    injection_metadata["added_context"].append(
                        f"[Table: {table}]"
                    )
            
            for col in detection_result["detected_columns"]:
                synonyms = self.scanner.get_column_synonyms(col)
                for syn in synonyms:
                    if syn not in self._injected_keywords:
                        self._injected_keywords.add(syn)
                        injection_metadata["added_context"].append(
                            f"[Column: {col} (aka: {syn})]"
                        )
        
        return enhanced_query, injection_metadata
    
    def get_injected_context(self, query: str) -> dict:
        """获取注入的上下文信息。
        
        Args:
            query: 用户查询
            
        Returns:
            包含表名、列名、建议表、建议列的字典
        """
        detection_result = self.scanner.detect_database_keywords(query)
        
        suggested_tables = detection_result["detected_tables"]
        suggested_columns = detection_result["detected_columns"]
        
        for table in suggested_tables:
            columns = self.scanner.get_table_column_mapping().get(table, [])
            suggested_columns.extend([c for c in columns if c not in suggested_columns])
        
        return {
            "suggested_tables": suggested_tables,
            "suggested_columns": suggested_columns,
            "table_column_mapping": self.scanner.get_table_column_mapping(),
            "detected_synonyms": detection_result["matched_synonyms"],
        }
    
    def should_use_hybrid_search(self, query: str) -> bool:
        """判断是否应该使用混合搜索。
        
        当查询中包含数据库关键词时返回True。
        
        Args:
            query: 用户查询
            
        Returns:
            是否应该使用混合搜索
        """
        detection_result = self.scanner.detect_database_keywords(query)
        return detection_result["has_database_reference"]
    
    def reset_injected_keywords(self):
        """重置已注入的关键词。"""
        self._injected_keywords.clear()
        logger.info("[DBKeywordInjector] Reset injected keywords")
    
    def get_all_injected_keywords(self) -> list[str]:
        """获取所有已注入的关键词。"""
        return list(self._injected_keywords)

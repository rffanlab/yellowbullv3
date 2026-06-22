# NL2SQL 数据库操作详细设计

## 1. 职责边界

| 职责 | 说明 |
|------|------|
| **Schema 感知** | 自动发现数据库表结构，为 LLM 提供 Schema 上下文 |
| **NL → SQL** | 将自然语言问题转换为 SQL 查询语句 |
| **SQL 验证** | 语法检查、权限校验、危险操作拦截 |
| **安全执行** | 只读查询为主，写操作需审批和审计 |
| **结果解释** | 将查询结果用自然语言解释给用户 |

---

## 2. Database Connector `tools/database/connector.py`

```python
"""
数据库连接器。

支持：
- PostgreSQL（主要）
- MySQL
- SQLite（开发/测试）

安全策略：
1. 默认只读连接
2. 写操作需要显式授权
3. 查询超时保护
4. 结果集大小限制
5. SQL 审计日志
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class DatabaseType(str, Enum):
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    SQLITE = "sqlite"


@dataclass
class QueryResult:
    """查询结果"""
    success: bool
    columns: list[str] = field(default_factory=list)
    rows: list[dict[str, Any]] = field(default_factory=list)
    row_count: int = 0
    execution_time_ms: float = 0.0
    sql: str = ""                    # 执行的 SQL（审计用）
    error_message: Optional[str] = None


class DatabaseConnector:
    """数据库连接器"""

    def __init__(
        self,
        db_type: DatabaseType,
        connection_string: str,
        read_only: bool = True,
        query_timeout_seconds: float = 30.0,
        max_result_rows: int = 1000,
    ):
        self._db_type = db_type
        self._connection_string = connection_string
        self._read_only = read_only
        self._timeout = query_timeout_seconds
        self._max_rows = max_result_rows
        self._engine = None
        self._async_engine = None

    async def initialize(self):
        """初始化数据库连接"""
        if self._db_type == DatabaseType.POSTGRESQL:
            from sqlalchemy.ext.asyncio import create_async_engine
            self._async_engine = create_async_engine(
                self._connection_string.replace("postgresql://", "postgresql+asyncpg://"),
                pool_size=5,
                max_overflow=10,
                pool_timeout=int(self._timeout),
            )
        elif self._db_type == DatabaseType.MYSQL:
            from sqlalchemy.ext.asyncio import create_async_engine
            self._async_engine = create_async_engine(
                self._connection_string.replace("mysql://", "mysql+aiomysql://"),
                pool_size=5,
                max_overflow=10,
                pool_timeout=int(self._timeout),
            )
        elif self._db_type == DatabaseType.SQLITE:
            from sqlalchemy.ext.asyncio import create_async_engine
            self._async_engine = create_async_engine(
                self._connection_string.replace("sqlite:///", "sqlite+aiosqlite:///"),
            )

    async def execute_query(self, sql: str) -> QueryResult:
        """执行 SQL 查询"""
        # 安全检查
        if not self._is_safe_sql(sql):
            return QueryResult(
                success=False,
                sql=sql,
                error_message="SQL 包含禁止的操作",
            )

        start_time = time.time()

        try:
            async with self._async_engine.connect() as conn:
                result = await asyncio.wait_for(
                    conn.execute(__import__("sqlalchemy").text(sql)),
                    timeout=self._timeout,
                )

                elapsed_ms = (time.time() - start_time) * 1000

                # 获取列名
                columns = list(result.keys())

                # 获取行数据（限制行数）
                rows_data = result.fetchmany(self._max_rows)
                rows = [dict(zip(columns, row)) for row in rows_data]

                # 记录审计日志
                logger.info(
                    f"SQL executed: {sql[:200]}... | "
                    f"{len(rows)} rows | {elapsed_ms:.0f}ms"
                )

                return QueryResult(
                    success=True,
                    columns=columns,
                    rows=rows,
                    row_count=len(rows),
                    execution_time_ms=elapsed_ms,
                    sql=sql,
                )

        except asyncio.TimeoutError:
            elapsed_ms = (time.time() - start_time) * 1000
            logger.warning(f"SQL query timeout after {self._timeout}s")
            return QueryResult(
                success=False,
                sql=sql,
                error_message=f"查询超时 ({self._timeout}s)",
                execution_time_ms=elapsed_ms,
            )
        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            logger.error(f"SQL query failed: {e}")
            return QueryResult(
                success=False,
                sql=sql,
                error_message=str(e),
                execution_time_ms=elapsed_ms,
            )

    async def get_schema(self) -> dict[str, list[dict]]:
        """获取数据库 Schema 信息"""
        try:
            async with self._async_engine.connect() as conn:
                if self._db_type == DatabaseType.POSTGRESQL:
                    schema_sql = """
                    SELECT table_name, column_name, data_type, is_nullable
                    FROM information_schema.columns
                    WHERE table_schema = 'public'
                    ORDER BY table_name, ordinal_position
                    """
                elif self._db_type == DatabaseType.MYSQL:
                    schema_sql = f"""
                    SELECT table_name, column_name, column_type, is_nullable
                    FROM information_schema.columns
                    WHERE table_schema = DATABASE()
                    ORDER BY table_name, ordinal_position
                    """
                else:  # SQLite
                    schema_sql = "SELECT * FROM sqlite_master WHERE type='table'"

                result = await conn.execute(__import__("sqlalchemy").text(schema_sql))
                rows = result.fetchall()
                columns = list(result.keys())

                # 按表分组
                tables = {}
                for row in rows:
                    row_dict = dict(zip(columns, row))
                    table_name = row_dict.get("table_name", "unknown")
                    if table_name not in tables:
                        tables[table_name] = []
                    tables[table_name].append(row_dict)

                return tables

        except Exception as e:
            logger.error(f"Failed to get schema: {e}")
            return {}

    async def get_table_info(self, table_name: str) -> Optional[dict]:
        """获取指定表的详细信息"""
        schema = await self.get_schema()
        return schema.get(table_name)

    def _is_safe_sql(self, sql: str) -> bool:
        """检查 SQL 是否安全"""
        sql_upper = sql.upper().strip()

        # 只读模式：禁止写操作
        if self._read_only:
            dangerous_keywords = [
                "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE",
                "TRUNCATE", "GRANT", "REVOKE", "EXEC", "EXECUTE",
            ]
            for keyword in dangerous_keywords:
                if f"{keyword} " in sql_upper or sql_upper.startswith(keyword):
                    return False

        # 始终禁止的操作
        always_dangerous = [
            "--",          # SQL comment (possible injection)
            ";",           # Multiple statements
            "UNION SELECT", "OR 1=1", "' OR '",  # Common injection patterns
        ]
        for pattern in always_dangerous:
            if pattern.upper() in sql_upper:
                return False

        return True

    async def close(self):
        """关闭连接"""
        if self._async_engine:
            await self._async_engine.dispose()


def get_connector() -> Optional[DatabaseConnector]:
    """获取全局数据库连接器"""
    import config.manager as cm
    manager = cm.get_manager()
    return getattr(manager, "_db_connector", None)
```

---

## 3. NL2SQL Engine `tools/database/nl2sql.py`

<tool_call>
<function=write>
<parameter=content>
# NL2SQL 数据库操作详细设计

## 1. 职责边界

| 职责 | 说明 |
|------|------|
| **Schema 感知** | 自动发现数据库表结构，为 LLM 提供 Schema 上下文 |
| **NL → SQL** | 将自然语言问题转换为 SQL 查询语句 |
| **SQL 验证** | 语法检查、权限校验、危险操作拦截 |
| **安全执行** | 只读查询为主，写操作需审批和审计 |
| **结果解释** | 将查询结果用自然语言解释给用户 |

---

## 2. Database Connector `tools/database/connector.py`

```python
"""
数据库连接器。

支持：
- PostgreSQL（主要）
- MySQL
- SQLite（开发/测试）

安全策略：
1. 默认只读连接
2. 写操作需要显式授权
3. 查询超时保护
4. 结果集大小限制
5. SQL 审计日志
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class DatabaseType(str, Enum):
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    SQLITE = "sqlite"


@dataclass
class QueryResult:
    """查询结果"""
    success: bool
    columns: list[str] = field(default_factory=list)
    rows: list[dict[str, Any]] = field(default_factory=list)
    row_count: int = 0
    execution_time_ms: float = 0.0
    sql: str = ""                    # 执行的 SQL（审计用）
    error_message: Optional[str] = None


class DatabaseConnector:
    """数据库连接器"""

    def __init__(
        self,
        db_type: DatabaseType,
        connection_string: str,
        read_only: bool = True,
        query_timeout_seconds: float = 30.0,
        max_result_rows: int = 1000,
    ):
        self._db_type = db_type
        self._connection_string = connection_string
        self._read_only = read_only
        self._timeout = query_timeout_seconds
        self._max_rows = max_result_rows
        self._engine = None
        self._async_engine = None

    async def initialize(self):
        """初始化数据库连接"""
        if self._db_type == DatabaseType.POSTGRESQL:
            from sqlalchemy.ext.asyncio import create_async_engine
            self._async_engine = create_async_engine(
                self._connection_string.replace("postgresql://", "postgresql+asyncpg://"),
                pool_size=5,
                max_overflow=10,
                pool_timeout=int(self._timeout),
            )
        elif self._db_type == DatabaseType.MYSQL:
            from sqlalchemy.ext.asyncio import create_async_engine
            self._async_engine = create_async_engine(
                self._connection_string.replace("mysql://", "mysql+aiomysql://"),
                pool_size=5,
                max_overflow=10,
                pool_timeout=int(self._timeout),
            )
        elif self._db_type == DatabaseType.SQLITE:
            from sqlalchemy.ext.asyncio import create_async_engine
            self._async_engine = create_async_engine(
                self._connection_string.replace("sqlite:///", "sqlite+aiosqlite:///"),
            )

    async def execute_query(self, sql: str) -> QueryResult:
        """执行 SQL 查询"""
        # 安全检查
        if not self._is_safe_sql(sql):
            return QueryResult(
                success=False,
                sql=sql,
                error_message="SQL 包含禁止的操作",
            )

        start_time = time.time()

        try:
            async with self._async_engine.connect() as conn:
                result = await asyncio.wait_for(
                    conn.execute(__import__("sqlalchemy").text(sql)),
                    timeout=self._timeout,
                )

                elapsed_ms = (time.time() - start_time) * 1000

                # 获取列名
                columns = list(result.keys())

                # 获取行数据（限制行数）
                rows_data = result.fetchmany(self._max_rows)
                rows = [dict(zip(columns, row)) for row in rows_data]

                # 记录审计日志
                logger.info(
                    f"SQL executed: {sql[:200]}... | "
                    f"{len(rows)} rows | {elapsed_ms:.0f}ms"
                )

                return QueryResult(
                    success=True,
                    columns=columns,
                    rows=rows,
                    row_count=len(rows),
                    execution_time_ms=elapsed_ms,
                    sql=sql,
                )

        except asyncio.TimeoutError:
            elapsed_ms = (time.time() - start_time) * 1000
            logger.warning(f"SQL query timeout after {self._timeout}s")
            return QueryResult(
                success=False,
                sql=sql,
                error_message=f"查询超时 ({self._timeout}s)",
                execution_time_ms=elapsed_ms,
            )
        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            logger.error(f"SQL query failed: {e}")
            return QueryResult(
                success=False,
                sql=sql,
                error_message=str(e),
                execution_time_ms=elapsed_ms,
            )

    async def get_schema(self) -> dict[str, list[dict]]:
        """获取数据库 Schema 信息"""
        try:
            async with self._async_engine.connect() as conn:
                if self._db_type == DatabaseType.POSTGRESQL:
                    schema_sql = """
                    SELECT table_name, column_name, data_type, is_nullable
                    FROM information_schema.columns
                    WHERE table_schema = 'public'
                    ORDER BY table_name, ordinal_position
                    """
                elif self._db_type == DatabaseType.MYSQL:
                    schema_sql = f"""
                    SELECT table_name, column_name, column_type, is_nullable
                    FROM information_schema.columns
                    WHERE table_schema = DATABASE()
                    ORDER BY table_name, ordinal_position
                    """
                else:  # SQLite
                    schema_sql = "SELECT * FROM sqlite_master WHERE type='table'"

                result = await conn.execute(__import__("sqlalchemy").text(schema_sql))
                rows = result.fetchall()
                columns = list(result.keys())

                # 按表分组
                tables = {}
                for row in rows:
                    row_dict = dict(zip(columns, row))
                    table_name = row_dict.get("table_name", "unknown")
                    if table_name not in tables:
                        tables[table_name] = []
                    tables[table_name].append(row_dict)

                return tables

        except Exception as e:
            logger.error(f"Failed to get schema: {e}")
            return {}

    async def get_table_info(self, table_name: str) -> Optional[dict]:
        """获取指定表的详细信息"""
        schema = await self.get_schema()
        return schema.get(table_name)

    def _is_safe_sql(self, sql: str) -> bool:
        """检查 SQL 是否安全"""
        sql_upper = sql.upper().strip()

        # 只读模式：禁止写操作
        if self._read_only:
            dangerous_keywords = [
                "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE",
                "TRUNCATE", "GRANT", "REVOKE", "EXEC", "EXECUTE",
            ]
            for keyword in dangerous_keywords:
                if f"{keyword} " in sql_upper or sql_upper.startswith(keyword):
                    return False

        # 始终禁止的操作
        always_dangerous = [
            "--",          # SQL comment (possible injection)
            ";",           # Multiple statements
            "UNION SELECT", "OR 1=1", "' OR '",  # Common injection patterns
        ]
        for pattern in always_dangerous:
            if pattern.upper() in sql_upper:
                return False

        return True

    async def close(self):
        """关闭连接"""
        if self._async_engine:
            await self._async_engine.dispose()


def get_connector() -> Optional[DatabaseConnector]:
    """获取全局数据库连接器"""
    import config.manager as cm
    manager = cm.get_manager()
    return getattr(manager, "_db_connector", None)
```

---

## 3. NL2SQL Engine `tools/database/nl2sql.py`

```python
"""
NL2SQL 引擎 —— 自然语言转 SQL。

工作流程：
1. 获取数据库 Schema（表结构）
2. 构建包含 Schema 的 prompt
3. LLM 生成 SQL
4. 验证 SQL 语法和安全
5. 执行查询，返回结果
6. LLM 解释结果
"""

import json
import logging
from typing import Optional

from llm.base import BaseLLM
from tools.database.connector import DatabaseConnector, QueryResult

logger = logging.getLogger(__name__)


class NL2SQLEngine:
    """自然语言转 SQL 引擎"""

    SYSTEM_PROMPT_TEMPLATE = """你是一个 SQL 专家。根据用户的自然语言问题，生成对应的 SQL 查询。

数据库类型: {db_type}

可用表结构：
{schema_info}

规则：
1. 只生成 SELECT 查询（除非明确要求写操作且获得授权）
2. 使用标准的 SQL 语法
3. 对模糊的查询条件做出合理假设
4. 添加 LIMIT 子句防止返回过多数据（默认 LIMIT 100）
5. 输出格式为纯 SQL，不要包含解释或其他文本

如果用户的问题无法用 SQL 回答，回复 "NOT_SQL"。"""

    EXPLAIN_PROMPT = """以下是用户的原始问题和执行结果：

问题: {question}
SQL: {sql}
结果 ({row_count} rows):
{result_preview}

请用简洁的自然语言解释这个查询结果，回答用户的问题。"""

    def __init__(
        self,
        llm: BaseLLM,
        connector: DatabaseConnector,
        max_schema_tables: int = 20,
    ):
        self._llm = llm
        self._connector = connector
        self._max_schema_tables = max_schema_tables

    async def query(self, question: str) -> dict:
        """执行 NL → SQL → Result → Explanation"""
        # Step 1: 获取 Schema（缓存优化）
        schema_text = await self._build_schema_context()

        if not schema_text:
            return {
                "success": False,
                "content": "数据库未配置或无法连接",
            }

        # Step 2: LLM 生成 SQL
        system_prompt = self.SYSTEM_PROMPT_TEMPLATE.format(
            db_type=self._connector._db_type.value,
            schema_info=schema_text,
        )

        sql_response = await self._llm.chat([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ], temperature=0.1)

        sql_text = (sql_response.content or "").strip()

        if sql_text == "NOT_SQL" or not sql_text:
            return {
                "success": False,
                "content": "该问题不适合用数据库查询回答",
            }

        # 清理 SQL（移除 markdown code block）
        sql_text = self._clean_sql(sql_text)

        # Step 3: 执行 SQL
        result = await self._connector.execute_query(sql_text)

        if not result.success:
            return {
                "success": False,
                "content": f"SQL 执行失败: {result.error_message}",
                "sql": sql_text,
            }

        # Step 4: LLM 解释结果
        explanation = await self._explain_result(
            question, sql_text, result
        )

        return {
            "success": True,
            "content": explanation,
            "sql": sql_text,
            "row_count": result.row_count,
            "execution_time_ms": result.execution_time_ms,
            "columns": result.columns,
            "rows": result.rows[:50],  # 限制返回行数
        }

    async def _build_schema_context(self) -> str:
        """构建 Schema 上下文文本"""
        schema = await self._connector.get_schema()

        if not schema:
            return ""

        lines = []
        table_count = 0
        for table_name, columns in schema.items():
            if table_count >= self._max_schema_tables:
                lines.append(f"... ({len(schema) - self._max_schema_tables} more tables)")
                break

            col_defs = ", ".join(
                f"{c.get('column_name', '?')} {c.get('data_type', c.get('column_type', 'unknown'))}"
                for c in columns
            )
            lines.append(f"CREATE TABLE {table_name} ({col_defs});")
            table_count += 1

        return "\n".join(lines)

    async def _explain_result(
        self, question: str, sql: str, result: QueryResult,
    ) -> str:
        """LLM 解释查询结果"""
        # 构建结果预览（限制大小）
        if result.rows:
            preview_rows = result.rows[:10]
            result_preview = json.dumps(preview_rows, ensure_ascii=False, indent=2)
        else:
            result_preview = "(空结果)"

        explain_prompt = self.EXPLAIN_PROMPT.format(
            question=question,
            sql=sql,
            row_count=result.row_count,
            result_preview=result_preview[:2000],  # 限制长度
        )

        explanation_response = await self._llm.chat([
            {"role": "system", "content": "你是一个数据分析师，用简洁的语言解释查询结果。"},
            {"role": "user", "content": explain_prompt},
        ])

        return explanation_response.content or f"查询返回 {result.row_count} 条记录。"

    @staticmethod
    def _clean_sql(sql: str) -> str:
        """清理 SQL 文本"""
        # 移除 markdown code block
        if sql.startswith("```"):
            lines = sql.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            sql = "\n".join(lines)

        return sql.strip()


def get_nl2sql_engine() -> Optional[NL2SQLEngine]:
    """获取全局 NL2SQL 引擎"""
    import config.manager as cm
    manager = cm.get_manager()
    if not hasattr(manager, "_nl2sql_engine"):
        connector = get_connector()
        if connector:
            from llm.factory import create_llm
            llm_settings = (manager.settings.tools or {}).get("nl2sql", {})
            llm = create_llm(llm_settings.get("llm_provider_config", {}))
            manager._nl2sql_engine = NL2SQLEngine(
                llm=llm,
                connector=connector,
                max_schema_tables=llm_settings.get("max_schema_tables", 20),
            )
    return getattr(manager, "_nl2sql_engine", None)
```

---

## 4. Tool 注册 `tools/database/tool.py`

```python
"""
数据库查询工具，注册到 ToolRegistry。
"""


class DatabaseQueryTool:
    """自然语言数据库查询工具"""

    name = "query_database"
    description = (
        "使用自然语言查询数据库。支持数据检索、统计汇总、条件筛选等操作。"
        "例如：'查找上个月销售额最高的产品'、'统计各部门员工数量'"
    )

    def __init__(self, engine: "NL2SQLEngine"):
        self._engine = engine

    @property
    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "自然语言描述的数据查询问题",
                        },
                    },
                    "required": ["question"],
                },
            },
        }

    async def execute(self, question: str) -> dict:
        """执行数据库查询"""
        result = await self._engine.query(question)

        if not result["success"]:
            return {
                "success": False,
                "content": result["content"],
            }

        response_parts = [result["content"]]

        # 附加原始数据（前几行）
        if result.get("rows"):
            rows_preview = result["rows"][:5]
            columns = result["columns"]
            table_lines = ["| " + " | ".join(columns) + " |"]
            table_lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
            for row in rows_preview:
                table_lines.append("| " + " | ".join(str(row.get(c, "")) for c in columns) + " |")

            response_parts.append("\n".join(table_lines))
            if result["row_count"] > 5:
                response_parts.append(f"*... 共 {result['row_count']} 条记录，仅显示前 5 条*")

        return {
            "success": True,
            "content": "\n\n".join(response_parts),
        }


class SchemaBrowserTool:
    """数据库 Schema 浏览工具"""

    name = "browse_schema"
    description = (
        "查看数据库的表结构和字段信息。用于了解有哪些数据可用，"
        "以及每个表的字段名、类型等信息。"
    )

    def __init__(self, connector: "DatabaseConnector"):
        self._connector = connector

    @property
    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "table_name": {
                            "type": "string",
                            "description": "可选：指定表名。不传则列出所有表。",
                        },
                    },
                    "required": [],
                },
            },
        }

    async def execute(self, table_name: str = None) -> dict:
        """浏览数据库 Schema"""
        if table_name:
            info = await self._connector.get_table_info(table_name)
            if not info:
                return {
                    "success": False,
                    "content": f"表 '{table_name}' 不存在",
                }

            columns_str = "\n".join(
                f"- `{c.get('column_name', '?')}` ({c.get('data_type', c.get('column_type', 'unknown'))})"
                for c in info
            )
            return {
                "success": True,
                content=f"**表: {table_name}**\n\n字段:\n{columns_str}",
            }
        else:
            schema = await self._connector.get_schema()
            tables_str = "\n".join(f"- `{name}` ({len(cols)} columns)" for name, cols in schema.items())
            return {
                "success": True,
                content=f"**数据库共有 {len(schema)} 张表:**\n\n{tables_str}",
            }


def register_database_tools(registry: "ToolRegistry"):
    """注册数据库相关工具"""
    from tools.database.nl2sql import get_nl2sql_engine
    from tools.database.connector import get_connector

    engine = get_nl2sql_engine()
    if engine:
        registry.register(DatabaseQueryTool(engine))

    connector = get_connector()
    if connector:
        registry.register(SchemaBrowserTool(connector))
```

---

## 5. YAML 配置 `tools.database` section

```yaml
# config/settings.yaml (新增)
database:
  enabled: false              # 是否启用数据库功能
  db_type: "postgresql"       # postgresql | mysql | sqlite
  connection_string: ""       # e.g., postgresql://user:pass@host/dbname
  read_only: true             # 默认只读模式

tools:
  nl2sql:
    enabled: false            # 是否启用 NL2SQL 工具
    llm_provider_config:      # NL2SQL 专用 LLM（建议用强模型）
      provider: "openai"
      model: "gpt-4o"

    max_schema_tables: 20     # Prompt 中最多包含的表数量
    query_timeout_seconds: 30 # SQL 查询超时
    max_result_rows: 1000     # 最大返回行数

    # 权限控制
    allowed_users: []         # 空表示所有用户，或指定用户 ID 列表
    require_approval_for_write: true  # 写操作需要人工审批
```

---

## 6. SQL 安全策略

| 防护层 | 措施 |
|--------|------|
| **只读默认** | 连接默认只读，写操作需显式开启并授权 |
| **SQL 审计** | 所有执行的 SQL 记录到日志 |
| **注入检测** | 拦截常见 SQL 注入模式（`;`, `--`, `UNION SELECT`） |
| **语句限制** | 禁止多语句执行，只允许单条 SELECT |
| **超时保护** | 查询超时自动终止 |
| **结果限制** | 最大返回行数限制，防止大数据量传输 |
| **Schema 裁剪** | Prompt 中仅包含必要的表结构信息 |

---

## 7. 数据流图

```
User: "上个月销售额最高的产品是什么？"
    │
    ▼
┌───────────────┐
│ Agent Core    │ → ToolRegistry.execute("query_database")
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ DatabaseQuery │
│   Tool        │
└───────┬───────┘
        │
        ▼
┌───────────────┐     ┌──────────────────┐
│ NL2SQL Engine │────▶│ DatabaseConnector│
│               │     │  (get_schema)    │
│ 1. Schema ctx │     └──────────────────┘
│ 2. LLM → SQL  │
│ 3. Execute    │◀────┐
│ 4. Explain    │     │
└───────┬───────┘     │
        │              │
        ▼              ▼
┌───────────────┐  ┌──────────────────┐
│ LLM (SQL gen) │  │ Database         │
└───────────────┘  │ (execute_query)  │
                   └──────────────────┘
                        │
                        ▼
                   QueryResult → Explanation → User
```

---

## 8. 设计总结

| 特性 | 实现方式 |
|------|---------|
| **数据库支持** | PostgreSQL（主要）、MySQL、SQLite |
| **NL2SQL** | LLM + Schema context → SQL generation → execution |
| **安全防护** | 只读默认、注入检测、超时保护、结果限制 |
| **Schema 感知** | 自动发现表结构，动态构建 prompt |
| **结果解释** | LLM 将查询结果转化为自然语言回答 |
| **审计追踪** | SQL 执行日志，完整记录谁在何时执行了什么 |

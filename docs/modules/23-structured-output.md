# 结构化输出解析详细设计（Structured Output）

## 1. 职责边界

| 职责 | 说明 |
|------|------|
| **Schema 定义** | 使用 Pydantic 模型定义期望的输出结构 |
| **LLM 约束生成** | 将 Schema 转换为 LLM 可理解的指令/JSON Schema |
| **输出解析** | 从 LLM 响应中提取并验证结构化数据 |
| **自动修复** | 对格式错误的输出进行重试或后处理修正 |

---

## 2. 协议设计 `structured_output/protocol.py`

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, TypeVar


T = TypeVar("T")


class ParseStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"       # 部分字段解析成功
    FAILED = "failed"         # 完全无法解析
    RETRYING = "retrying"     # 正在重试修复


@dataclass(frozen=True)
class ParseError:
    """解析错误详情"""
    error_type: str           # json_decode | schema_validation | type_mismatch
    message: str              # 错误描述
    raw_output: str           # LLM 原始输出（用于调试/重试）
    line_number: int | None = None


@dataclass(frozen=True)
class StructuredResult(Generic[T]):
    """结构化解析结果"""
    status: ParseStatus       # 解析状态
    data: T | None = None     # 解析后的数据
    errors: list[ParseError] = field(default_factory=list)  # 错误列表
    raw_output: str = ""      # LLM 原始输出
    confidence: float = 0.0   # 解析置信度 [0, 1]


@dataclass(frozen=True)
class OutputSchema:
    """输出 Schema 定义"""
    name: str                 # Schema 名称，如 "code_review_result"
    description: str          # 人类可读描述
    json_schema: dict[str, Any]   # JSON Schema 格式
    pydantic_model: type | None = None  # Pydantic 模型类（可选）
    required_fields: list[str] = field(default_factory=list)  # 必填字段
```

---

## 3. Schema 构建器 `structured_output/schema_builder.py`

```python
"""
Schema 构建模块。

将 Pydantic 模型或字典定义转换为 LLM 可理解的输出约束指令。
"""

import json
import logging
from typing import Any, Type

try:
    from pydantic import BaseModel
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False


logger = logging.getLogger(__name__)


class SchemaBuilder:
    """
    输出 Schema 构建器。

    支持两种输入方式：
    1. Pydantic 模型（推荐）→ 自动提取 JSON Schema + 字段描述
    2. 字典定义 → 手动指定结构
    """

    @classmethod
    def from_pydantic(cls, model: Type["BaseModel"]) -> "OutputSchema":
        """从 Pydantic 模型构建 Schema"""
        if not HAS_PYDANTIC:
            raise ImportError("Pydantic is required for this feature")

        schema = model.model_json_schema()
        name = schema.get("title", model.__name__)
        description = schema.get("description", "")

        # 提取必填字段
        required = list(schema.get("required", []))

        return OutputSchema(
            name=name,
            description=description or f"Structured output for {name}",
            json_schema=schema,
            pydantic_model=model,
            required_fields=required,
        )

    @classmethod
    def from_dict(cls, definition: dict[str, Any]) -> "OutputSchema":
        """从字典定义构建 Schema"""
        return OutputSchema(
            name=definition.get("name", "custom_output"),
            description=definition.get("description", ""),
            json_schema=definition.get("schema", {}),
            required_fields=definition.get("required_fields", []),
        )

    @staticmethod
    def to_llm_instruction(schema: OutputSchema) -> str:
        """
        将 Schema 转换为 LLM 指令。

        返回的字符串可直接注入 system prompt，约束 LLM 输出格式。
        """
        schema_json = json.dumps(schema.json_schema, ensure_ascii=False, indent=2)

        instruction = (
            f"请严格按照以下 JSON Schema 格式返回结果：\n\n"
            f"```json\n{schema_json}\n```\n\n"
            f"要求：\n"
            f"1. 只返回合法的 JSON，不要包含任何其他文本或解释\n"
            f"2. 所有必填字段必须提供值\n"
            f"3. 枚举类型的值必须在允许范围内\n"
            f"4. 如果某个字段不适用，使用 null（不可省略）\n"
        )

        if schema.required_fields:
            instruction += (
                f"\n必填字段：{', '.join(schema.required_fields)}\n"
            )

        return instruction


class CommonSchemas:
    """预定义的常用输出 Schema"""

    @staticmethod
    def code_review() -> OutputSchema:
        """代码审查结果 Schema"""
        if HAS_PYDantic:
            class CodeReviewResult(BaseModel):
                """代码审查结果"""
                overall_score: float = ...  # 总体评分 [0, 10]
                summary: str = ...          # 审查摘要
                issues: list[dict[str, Any]] = field(  # type: ignore
                    default_factory=list
                )  # 问题列表 [{"severity": "high", "line": 42, "message": "..."}]
                suggestions: list[str] = field(default_factory=list)  # 改进建议
                is_merge_ready: bool = ...  # 是否可合并

            return SchemaBuilder.from_pydantic(CodeReviewResult)

        return SchemaBuilder.from_dict({
            "name": "code_review_result",
            "description": "代码审查结果",
            "schema": {
                "type": "object",
                "properties": {
                    "overall_score": {"type": "number", "minimum": 0, "maximum": 10},
                    "summary": {"type": "string"},
                    "issues": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "severity": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
                                "line": {"type": "integer"},
                                "message": {"type": "string"},
                            },
                        },
                    },
                    "suggestions": {"type": "array", "items": {"type": "string"}},
                    "is_merge_ready": {"type": "boolean"},
                },
                "required": ["overall_score", "summary", "issues", "is_merge_ready"],
            },
        })

    @staticmethod
    def sentiment_analysis() -> OutputSchema:
        """情感分析结果 Schema"""
        return SchemaBuilder.from_dict({
            "name": "sentiment_result",
            "description": "文本情感分析结果",
            "schema": {
                "type": "object",
                "properties": {
                    "sentiment": {"type": "string", "enum": ["positive", "neutral", "negative"]},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "key_phrases": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["sentiment", "confidence"],
            },
        })

    @staticmethod
    def entity_extraction() -> OutputSchema:
        """实体提取结果 Schema"""
        return SchemaBuilder.from_dict({
            "name": "entity_extraction_result",
            "description": "从文本中提取结构化实体",
            "schema": {
                "type": "object",
                "properties": {
                    "entities": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string"},
                                "label": {"type": "string"},
                                "start_pos": {"type": "integer"},
                                "end_pos": {"type": "integer"},
                            },
                        },
                    },
                },
                "required": ["entities"],
            },
        })
```

---

## 4. 解析引擎 `structured_output/parser.py`

```python
"""
结构化输出解析器。

核心流程：
1. 提取 LLM 响应中的 JSON 部分（处理 markdown 包裹、多余文本）
2. 使用 Pydantic / jsonschema 验证结构
3. 失败时自动重试或后处理修复
"""

import json
import logging
import re
from typing import Any, TypeVar, Generic

logger = logging.getLogger(__name__)

T = TypeVar("T")


class StructuredParser(Generic[T]):
    """
    结构化输出解析器。

    支持：
    - JSON 提取（处理 markdown code block、前后多余文本）
    - Pydantic 模型验证
    - jsonschema 验证
    - 自动重试修复
    """

    # 匹配 JSON 代码块或裸 JSON
    JSON_PATTERNS = [
        re.compile(r"```(?:json)?\s*\n([\s\S]*?)\n?```"),   # ```json ... ```
        re.compile(r"\{[\s\S]*\}"),                           # 裸 JSON object
        re.compile(r"\[[\s\S]*\]"),                           # 裸 JSON array
    ]

    def __init__(
        self,
        schema: "OutputSchema",
        max_retries: int = 2,
        llm_client: Any | None = None,
    ):
        self._schema = schema
        self._max_retries = max_retries
        self._llm = llm_client

    async def parse(self, raw_output: str) -> StructuredResult[T]:
        """
        解析 LLM 输出为结构化数据。

        Args:
            raw_output: LLM 原始响应文本

        Returns:
            StructuredResult，包含解析状态和数据
        """
        # Step 1: 提取 JSON
        json_str = self._extract_json(raw_output)
        if not json_str:
            return StructuredResult(
                status=ParseStatus.FAILED,
                errors=[ParseError("json_decode", "无法找到 JSON 内容", raw_output)],
                raw_output=raw_output,
            )

        # Step 2: 解析 JSON
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode failed: {e}, attempting repair")
            return await self._attempt_repair(raw_output, json_str)

        # Step 3: 验证 Schema
        validation_errors = self._validate(data)
        if not validation_errors:
            parsed_data = self._cast_to_model(data)
            return StructuredResult(
                status=ParseStatus.SUCCESS,
                data=parsed_data,
                raw_output=raw_output,
                confidence=self._calculate_confidence(data),
            )

        # Step 4: 验证失败，尝试修复
        logger.warning(f"Schema validation failed: {validation_errors}")
        return await self._attempt_repair(raw_output, json_str)

    def _extract_json(self, text: str) -> str | None:
        """从文本中提取 JSON 字符串"""
        for pattern in self.JSON_PATTERNS:
            match = pattern.search(text)
            if match:
                return match.group(1) if match.lastindex else match.group(0)
        return None

    def _validate(self, data: Any) -> list[str]:
        """验证数据是否符合 Schema"""
        errors = []

        # Pydantic 模型验证
        if self._schema.pydantic_model and HAS_PYDANTIC:
            try:
                self._schema.pydantic_model.model_validate(data)
                return errors
            except Exception as e:
                errors.append(str(e))
                return errors

        # JSON Schema 验证（简化实现）
        schema = self._schema.json_schema
        if schema.get("type") == "object" and isinstance(data, dict):
            for field in self._schema.required_fields:
                if field not in data:
                    errors.append(f"Missing required field: {field}")

            properties = schema.get("properties", {})
            for field_name, field_schema in properties.items():
                if field_name in data:
                    value = data[field_name]
                    expected_type = field_schema.get("type")
                    if expected_type and not self._check_type(value, expected_type):
                        errors.append(
                            f"Field '{field_name}' expected type {expected_type}, "
                            f"got {type(value).__name__}"
                        )

        return errors

    def _check_type(self, value: Any, json_type: str) -> bool:
        """检查值是否符合 JSON Schema 类型"""
        type_map = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
            "null": type(None),
        }
        expected = type_map.get(json_type)
        if not expected:
            return True
        return isinstance(value, expected)

    def _cast_to_model(self, data: Any) -> T:
        """将字典转换为 Pydantic 模型实例"""
        if self._schema.pydantic_model and HAS_PYDANTIC:
            return self._schema.pydantic_model.model_validate(data)  # type: ignore
        return data  # type: ignore

    def _calculate_confidence(self, data: Any) -> float:
        """计算解析置信度"""
        if not isinstance(data, dict):
            return 0.8

        total = len(self._schema.required_fields)
        present = sum(1 for f in self._schema.required_fields if f in data and data[f] is not None)
        return present / max(total, 1)

    async def _attempt_repair(
        self, raw_output: str, broken_json: str | None
    ) -> StructuredResult[T]:
        """
        尝试修复解析失败的输出。

        策略：
        1. 简单后处理（移除尾逗号、补全括号等）
        2. LLM 辅助修复（如果有 LLM client）
        """
        # 简单后处理
        if broken_json:
            repaired = self._simple_repair(broken_json)
            try:
                data = json.loads(repaired)
                validation_errors = self._validate(data)
                if not validation_errors:
                    return StructuredResult(
                        status=ParseStatus.PARTIAL,
                        data=self._cast_to_model(data),
                        raw_output=raw_output,
                        confidence=0.5,
                    )
            except json.JSONDecodeError:
                pass

        # LLM 辅助修复
        if self._llm and self._max_retries > 0:
            return await self._repair_with_llm(raw_output)

        return StructuredResult(
            status=ParseStatus.FAILED,
            errors=[ParseError("schema_validation", "无法修复输出格式", raw_output)],
            raw_output=raw_output,
        )

    def _simple_repair(self, json_str: str) -> str:
        """简单 JSON 修复"""
        # 移除尾逗号（对象/数组末尾的逗号）
        json_str = re.sub(r",\s*([}\]])", r"\1", json_str)
        # 移除行尾注释
        json_str = "\n".join(
            line for line in json_str.split("\n") if not line.strip().startswith("//")
        )
        return json_str

    async def _repair_with_llm(self, raw_output: str) -> StructuredResult[T]:
        """使用 LLM 修复格式错误的输出"""
        repair_prompt = (
            f"以下 JSON 格式有误，请修正为合法 JSON。只返回修正后的 JSON，不要其他内容。\n\n"
            f"期望的 Schema：\n{json.dumps(self._schema.json_schema, ensure_ascii=False, indent=2)}\n\n"
            f"原始输出：\n{raw_output}"
        )

        try:
            repaired = await self._llm.generate(repair_prompt)
            json_str = self._extract_json(repaired)
            if not json_str:
                return StructuredResult(
                    status=ParseStatus.FAILED,
                    errors=[ParseError("repair_failed", "LLM 修复失败", raw_output)],
                    raw_output=raw_output,
                )

            data = json.loads(json_str)
            parsed_data = self._cast_to_model(data)

            return StructuredResult(
                status=ParseStatus.PARTIAL,
                data=parsed_data,
                raw_output=raw_output,
                confidence=0.4,
            )
        except Exception as e:
            logger.error(f"LLM repair failed: {e}")
            return StructuredResult(
                status=ParseStatus.FAILED,
                errors=[ParseError("repair_failed", str(e), raw_output)],
                raw_output=raw_output,
            )


# 兼容无 Pydantic 环境
try:
    from pydantic import BaseModel as _BM
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False
```

---

## 5. Agent 集成 `structured_output/integration.py`

```python
"""
结构化输出与 Agent 核心的集成。

在 LLM 调用层注入 Schema 约束，在响应处理层解析结果。
"""

import logging
from typing import Any, TypeVar

from structured_output.parser import StructuredParser, ParseStatus, StructuredResult
from structured_output.schema_builder import OutputSchema, SchemaBuilder

logger = logging.getLogger(__name__)

T = TypeVar("T")


class StructuredOutputManager:
    """
    结构化输出管理器。

    使用方式：
    ```python
    manager = StructuredOutputManager(llm_client)

    # 定义期望的输出结构
    schema = CommonSchemas.code_review()

    # 调用 LLM 并获取结构化结果
    result = await manager.generate(
        prompt="审查以下代码：\n\n" + code,
        schema=schema,
    )

    if result.status == ParseStatus.SUCCESS:
        print(f"Score: {result.data.overall_score}")
    ```
    """

    def __init__(self, llm_client: Any):
        self._llm = llm_client

    async def generate(
        self,
        prompt: str,
        schema: OutputSchema,
        system_prompt: str | None = None,
        max_retries: int = 2,
    ) -> StructuredResult[T]:
        """
        生成结构化输出。

        Args:
            prompt:      用户提示词
            schema:      期望的输出 Schema
            system_prompt: 自定义 system prompt（可选）
            max_retries: 最大重试次数

        Returns:
            StructuredResult[T]
        """
        # 构建带 Schema 约束的 system prompt
        schema_instruction = SchemaBuilder.to_llm_instruction(schema)

        full_system = (system_prompt or "你是一个专业的 AI 助手。") + "\n\n" + schema_instruction

        parser = StructuredParser(
            schema=schema,
            max_retries=max_retries,
            llm_client=self._llm,
        )

        # 调用 LLM
        raw_output = await self._llm.generate(
            messages=[
                {"role": "system", "content": full_system},
                {"role": "user", "content": prompt},
            ]
        )

        return await parser.parse(raw_output)


class ToolOutputParser:
    """
    工具输出结构化解析器。

    用于将工具的返回结果转换为 Agent 可理解的结构化格式，
    便于后续的条件判断、数据提取等操作。
    """

    def __init__(self, llm_client: Any):
        self._llm = llm_client
        self._manager = StructuredOutputManager(llm_client)

    async def parse_tool_output(
        self,
        tool_name: str,
        raw_output: str,
        expected_schema: OutputSchema,
    ) -> StructuredResult[T]:
        """解析工具输出为结构化数据"""
        prompt = (
            f"以下工具 '{tool_name}' 的输出如下，请将其转换为结构化格式：\n\n"
            f"{raw_output}"
        )

        return await self._manager.generate(prompt, expected_schema)
```

---

## 6. 配置项 `config/structured_output.yaml`

```yaml
structured_output:
  enabled: true                       # 是否启用结构化输出解析
  max_retries: 2                      # 最大重试次数
  use_llm_for_repair: true            # 失败时使用 LLM 修复
  repair_model: "gpt-4o-mini"         # 修复使用的模型（低成本）

  validation:
    strict_mode: false                # 严格模式：缺失必填字段直接报错
    auto_cast_types: true             # 自动类型转换（如字符串数字 → int）
```

---

## 7. 与现有模块的交互

| 模块 | 交互方式 |
|------|---------|
| `02-llm-abstraction` | Schema 约束注入到 LLM 调用的 system prompt |
| `03-tool-system` | 工具返回结果通过 StructuredParser 解析 |
| `15-multi-agent-collaboration` | Agent 间通信使用结构化消息格式 |

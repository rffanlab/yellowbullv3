# 主动追问机制详细设计（Proactive Follow-up）

## 1. 职责边界

| 职责 | 说明 |
|------|------|
| **信息不足检测** | 识别用户输入中缺失的关键信息，判断是否需要追问 |
| **澄清问题生成** | 根据上下文自动生成精准、友好的追问内容 |
| **追问策略控制** | 管理追问次数上限、追问时机、追问优先级 |
| **多轮收敛** | 通过多轮追问逐步收集完整信息，最终执行用户意图 |

---

## 2. 协议设计 `followup/protocol.py`

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class FollowUpType(str, Enum):
    CLARIFICATION = "clarification"   # 澄清模糊意图
    MISSING_PARAM = "missing_param"   # 缺少必要参数
    CONFIRMATION = "confirmation"     # 确认高风险操作
    PREFERENCE = "preference"         # 了解用户偏好


@dataclass(frozen=True)
class MissingInfo:
    """缺失信息描述"""
    field_name: str                   # 缺失字段名，如 "target_date"
    description: str                  # 人类可读的描述
    is_required: bool = True          # 是否必填
    suggested_values: list[str] | None = None  # 建议值列表


@dataclass(frozen=True)
class FollowUpRequest:
    """追问请求"""
    follow_up_type: FollowUpType      # 追问类型
    question: str                     # 生成的追问内容
    missing_info: list[MissingInfo]   # 缺失信息列表
    context_summary: str | None = None  # 上下文摘要（用于 LLM 生成问题）
    priority: int = 1                 # 优先级 [1-5], 5 最高


@dataclass
class FollowUpState:
    """追问状态跟踪"""
    session_id: str                   # 会话 ID
    original_intent: str | None = None  # 原始识别的意图
    collected_info: dict[str, Any] = field(default_factory=dict)  # 已收集信息
    pending_questions: list[FollowUpRequest] = field(default_factory=list)  # 待回答追问
    follow_up_count: int = 0          # 已追问次数
    max_follow_ups: int = 3           # 最大追问次数
    is_resolved: bool = False         # 是否已收集到足够信息


@dataclass(frozen=True)
class FollowUpResponse:
    """用户对追问的回复"""
    session_id: str                   # 会话 ID
    answer: str                       # 用户回答内容
    answered_fields: list[str] | None = None  # 回答了哪些字段
```

---

## 3. 信息不足检测器 `followup/detector.py`

```python
"""
信息不足检测模块。

通过 LLM 分析用户输入，判断是否缺少执行意图所需的关键信息。
"""

import json
import logging
from typing import Any, Optional

from followup.protocol import FollowUpType, MissingInfo

logger = logging.getLogger(__name__)


class InsufficientInfoDetector:
    """
    信息不足检测器。

    工作流程：
    1. 接收用户输入和已识别的意图/工具
    2. 分析该意图所需的全部参数
    3. 对比用户输入，找出缺失的必要参数
    4. 返回缺失信息列表
    """

    DETECTION_PROMPT = """你是一个意图分析助手。请分析用户的请求是否包含足够的信息来执行操作。

已知工具/操作的必要参数：
{tool_params}

用户输入：
{user_input}

对话历史（最近 5 轮）：
{history}

请以 JSON 格式返回分析结果：
{{
  "sufficient": true/false,           // 信息是否充足
  "missing_fields": [                // 缺失的必要字段
    {{
      "field_name": "...",
      "description": "人类可读的描述",
      "is_required": true/false,
      "suggested_values": ["...", "..."]  // 可选，给出建议值
    }}
  ],
  "follow_up_type": "clarification" | "missing_param" | "confirmation" | "preference",
  "confidence": 0.95                 // 判断置信度
}}"""

    def __init__(self, llm_client: Any):
        self._llm = llm_client

    async def detect(
        self,
        user_input: str,
        tool_params: dict[str, Any],
        history: list[dict[str, str]] | None = None,
    ) -> tuple[bool, list[MissingInfo], FollowUpType]:
        """
        检测信息是否充足。

        Args:
            user_input:   用户当前输入
            tool_params:  目标工具/操作的参数定义
            history:      最近对话历史

        Returns:
            (is_sufficient, missing_info_list, follow_up_type)
        """
        prompt = self.DETECTION_PROMPT.format(
            tool_params=json.dumps(tool_params, ensure_ascii=False, indent=2),
            user_input=user_input,
            history=self._format_history(history or []),
        )

        try:
            response = await self._llm.generate(prompt)
            result = json.loads(response.strip())

            is_sufficient = result.get("sufficient", False)
            follow_up_type = FollowUpType(
                result.get("follow_up_type", "clarification")
            )

            missing_info = [
                MissingInfo(**mi)
                for mi in result.get("missing_fields", [])
            ]

            return is_sufficient, missing_info, follow_up_type

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse detection result: {e}")
            # Fallback: 假设信息不足，需要追问
            return False, [], FollowUpType.CLARIFICATION

    def _format_history(self, history: list[dict[str, str]]) -> str:
        """格式化对话历史为可读字符串"""
        if not history:
            return "(无)"
        lines = []
        for turn in history[-5:]:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            lines.append(f"[{role}]: {content}")
        return "\n".join(lines)


class HighRiskDetector:
    """
    高风险操作检测器。

    识别需要用户确认的高风险操作，如：
    - 数据库写操作（DELETE, UPDATE）
    - 文件删除/覆盖
    - 外部 API 调用（付费服务）
    - 权限变更操作
    """

    HIGH_RISK_PATTERNS = [
        ("delete", "删除"),
        ("drop", "删除"),
        ("truncate", "清空"),
        ("remove", "移除"),
        ("overwrite", "覆盖"),
        ("execute", "执行"),
        ("send_payment", "支付"),
        ("transfer", "转账"),
    ]

    @staticmethod
    def is_high_risk(tool_name: str, arguments: dict[str, Any]) -> bool:
        """判断操作是否属于高风险"""
        tool_lower = tool_name.lower()
        for pattern, _ in HighRiskDetector.HIGH_RISK_PATTERNS:
            if pattern in tool_lower:
                return True

        # 检查参数中的敏感值
        for value in arguments.values():
            if isinstance(value, str):
                val_lower = value.lower()
                for pattern, _ in HighRiskDetector.HIGH_RISK_PATTERNS:
                    if pattern in val_lower:
                        return True

        return False

    @staticmethod
    def get_risk_description(tool_name: str) -> str:
        """获取风险描述"""
        for pattern, desc in HighRiskDetector.HIGH_RISK_PATTERNS:
            if pattern in tool_name.lower():
                return f"此操作涉及{desc}，请确认是否继续。"
        return "此操作可能产生不可逆的影响，请确认是否继续。"
```

---

## 4. 追问生成器 `followup/generator.py`

```python
"""
追问内容生成模块。

根据缺失信息类型和上下文，生成自然、友好的追问内容。
"""

import logging
from typing import Any

from followup.protocol import FollowUpRequest, FollowUpType, MissingInfo

logger = logging.getLogger(__name__)


class FollowUpGenerator:
    """
    追问生成器。

    策略：
    - 单次最多追问 2 个问题，避免给用户造成压力
    - 优先追问必填字段
    - 有建议值时提供选项，降低用户输入成本
    - 语气友好、简洁
    """

    GENERATION_PROMPT = """你是一个友好的 AI 助手。用户提出了一个请求，但信息不完整。
请根据以下信息生成自然的追问内容。

缺失的信息：
{missing_info}

已有上下文：
{context}

追问要求：
1. 最多问 2 个问题
2. 语气友好、简洁
3. 如果有建议值，用选项形式呈现（如："您希望设置为 A、B 还是 C？"）
4. 不要解释为什么要追问，直接提问即可

请只返回追问内容，不要其他说明。"""

    def __init__(self, llm_client: Any):
        self._llm = llm_client

    async def generate(
        self,
        missing_info: list[MissingInfo],
        follow_up_type: FollowUpType,
        context_summary: str | None = None,
    ) -> FollowUpRequest:
        """生成追问请求"""
        if not missing_info:
            question = "能否提供更多细节，以便我更好地帮助您？"
        else:
            question = await self._generate_question(
                missing_info, follow_up_type, context_summary
            )

        return FollowUpRequest(
            follow_up_type=follow_up_type,
            question=question,
            missing_info=missing_info[:2],  # 最多追问 2 个字段
            context_summary=context_summary,
            priority=self._calculate_priority(missing_info, follow_up_type),
        )

    async def _generate_question(
        self,
        missing_info: list[MissingInfo],
        follow_up_type: FollowUpType,
        context: str | None,
    ) -> str:
        """使用 LLM 生成追问内容"""
        info_text = "\n".join(
            f"- {m.field_name}: {m.description}"
            + (f"（建议值：{', '.join(m.suggested_values)}）" if m.suggested_values else "")
            for m in missing_info
        )

        prompt = self.GENERATION_PROMPT.format(
            missing_info=info_text,
            context=context or "(无)",
        )

        try:
            response = await self._llm.generate(prompt)
            return response.strip()
        except Exception as e:
            logger.warning(f"Failed to generate follow-up question: {e}")
            # Fallback 模板
            return self._template_question(missing_info, follow_up_type)

    def _template_question(
        self,
        missing_info: list[MissingInfo],
        follow_up_type: FollowUpType,
    ) -> str:
        """模板化追问（LLM 不可用时的 fallback）"""
        if follow_up_type == FollowUpType.CONFIRMATION:
            return "请确认是否继续执行此操作？"

        required = [m for m in missing_info if m.is_required]
        if not required:
            return "能否提供更多细节，以便我更好地帮助您？"

        fields = ", ".join(m.description for m in required[:2])
        return f"请问：{fields}？"

    def _calculate_priority(
        self,
        missing_info: list[MissingInfo],
        follow_up_type: FollowUpType,
    ) -> int:
        """计算追问优先级"""
        if follow_up_type == FollowUpType.CONFIRMATION:
            return 5  # 安全确认最高优先

        required_count = sum(1 for m in missing_info if m.is_required)
        if required_count > 0:
            return min(4, 2 + required_count)

        return 1


class FollowUpStateTracker:
    """
    追问状态跟踪器。

    管理多轮追问过程中的信息收集进度，
    防止无限追问循环。
    """

    def __init__(self):
        self._states: dict[str, Any] = {}  # session_id -> FollowUpState

    def init_state(
        self,
        session_id: str,
        original_intent: str | None = None,
        max_follow_ups: int = 3,
    ) -> "FollowUpStateTracker":
        """初始化追问状态"""
        from followup.protocol import FollowUpState

        self._states[session_id] = FollowUpState(
            session_id=session_id,
            original_intent=original_intent,
            max_follow_ups=max_follow_ups,
        )
        return self

    def get_state(self, session_id: str) -> Any | None:
        """获取会话的追问状态"""
        return self._states.get(session_id)

    def can_follow_up(self, session_id: str) -> bool:
        """判断是否还能继续追问"""
        state = self._states.get(session_id)
        if not state:
            return True
        return state.follow_up_count < state.max_follow_ups and not state.is_resolved

    def record_answer(
        self, session_id: str, answered_fields: dict[str, Any]
    ) -> bool:
        """
        记录用户回答，更新已收集信息。

        Returns:
            True 如果已收集到足够信息
        """
        state = self._states.get(session_id)
        if not state:
            return False

        state.collected_info.update(answered_fields)
        state.follow_up_count += 1

        # TODO: 根据意图的参数定义判断是否已收集完整
        # 这里简化为：追问次数达到上限或明确标记为 resolved
        if state.follow_up_count >= state.max_follow_ups:
            state.is_resolved = True

        return state.is_resolved

    def cleanup(self, session_id: str) -> None:
        """清理会话的追问状态"""
        self._states.pop(session_id, None)
```

---

## 5. 集成中间件 `followup/middleware.py`

```python
"""
主动追问中间件。

挂载在 Agent 消息处理管道中，位于意图识别之后、工具执行之前。
"""

import logging
from typing import Any, Optional

from followup.detector import HighRiskDetector, InsufficientInfoDetector
from followup.generator import FollowUpGenerator, FollowUpStateTracker
from followup.protocol import (
    FollowUpRequest,
    FollowUpResponse,
    FollowUpType,
)

logger = logging.getLogger(__name__)


class FollowUpMiddleware:
    """
    主动追问中间件。

    处理流程：
    1. 意图识别后，检查信息是否充足
    2. 如果不足，生成追问并返回给用户（中断工具执行）
    3. 收到用户回答后，解析并更新已收集信息
    4. 重复直到信息充足或达到追问上限
    """

    def __init__(
        self,
        detector: InsufficientInfoDetector,
        generator: FollowUpGenerator,
        tracker: FollowUpStateTracker,
    ):
        self._detector = detector
        self._generator = generator
        self._tracker = tracker

    async def check_and_follow_up(
        self,
        session_id: str,
        user_input: str,
        tool_name: str | None,
        tool_params: dict[str, Any],
        provided_args: dict[str, Any],
        history: list[dict[str, str]] | None = None,
    ) -> FollowUpRequest | None:
        """
        检查是否需要追问。

        Args:
            session_id:     会话 ID
            user_input:     用户输入
            tool_name:      识别到的目标工具名
            tool_params:    工具的参数定义
            provided_args:  用户已提供的参数
            history:        对话历史

        Returns:
            FollowUpRequest 如果需要追问，否则 None
        """
        if not self._tracker.can_follow_up(session_id):
            logger.info(f"Max follow-ups reached for session {session_id}")
            return None

        # 高风险操作检测
        if HighRiskDetector.is_high_risk(tool_name or "", provided_args):
            risk_desc = HighRiskDetector.get_risk_description(tool_name or "")
            return FollowUpRequest(
                follow_up_type=FollowUpType.CONFIRMATION,
                question=f"{risk_desc}",
                missing_info=[],
                priority=5,
            )

        # 信息充足性检测
        is_sufficient, missing_info, follow_up_type = await self._detector.detect(
            user_input, tool_params, history
        )

        if is_sufficient:
            return None

        if not missing_info:
            logger.info(f"Info sufficient for session {session_id}")
            return None

        # 初始化追问状态
        self._tracker.init_state(session_id, original_intent=tool_name)

        # 生成追问
        request = await self._generator.generate(
            missing_info, follow_up_type
        )

        logger.info(
            f"Generated follow-up for session {session_id}: "
            f"type={follow_up_type}, priority={request.priority}"
        )
        return request

    async def process_answer(
        self,
        session_id: str,
        answer: str,
        expected_fields: list[str],
    ) -> dict[str, Any]:
        """
        处理用户对追问的回答。

        Args:
            session_id:     会话 ID
            answer:         用户回答内容
            expected_fields: 期望收集的字段名列表

        Returns:
            解析出的参数字典
        """
        # TODO: 使用 LLM 从自由文本中提取结构化参数
        parsed = {}

        for field_name in expected_fields:
            # 简化实现：直接映射
            parsed[field_name] = answer

        is_resolved = self._tracker.record_answer(session_id, parsed)

        if is_resolved:
            logger.info(f"Follow-up resolved for session {session_id}")
            self._tracker.cleanup(session_id)

        return parsed

    def should_interrupt(self, request: FollowUpRequest | None) -> bool:
        """判断是否应中断当前执行流程进行追问"""
        if not request:
            return False
        # 高优先级追问总是中断
        return request.priority >= 3
```

---

## 6. Agent 核心集成示例

```python
"""
在 Agent 核心中的使用方式。
"""

# 在 agent 主循环中：
async def process_message(agent, session_id: str, user_input: str):
    # 1. 意图识别 + 工具选择
    intent_result = await agent.intent_classifier.classify(user_input)
    tool_name = intent_result.tool_name
    tool_params = agent.tool_registry.get_params(tool_name)

    # 2. 参数提取
    provided_args = await agent.extract_parameters(user_input, tool_params)

    # 3. 【追问检查】信息不足时中断并追问
    follow_up = await agent.followup_middleware.check_and_follow_up(
        session_id=session_id,
        user_input=user_input,
        tool_name=tool_name,
        tool_params=tool_params,
        provided_args=provided_args,
        history=agent.session_manager.get_history(session_id),
    )

    if agent.followup_middleware.should_interrupt(follow_up):
        # 返回追问内容，等待用户回答
        return await agent.respond(session_id, follow_up.question)

    # 4. 信息充足，执行工具调用
    result = await agent.tool_executor.execute(tool_name, provided_args)
    return await agent.respond(session_id, format_result(result))


# 收到用户追问回复时：
async def process_followup_answer(agent, session_id: str, answer: str):
    state = agent.followup_middleware._tracker.get_state(session_id)
    expected = [m.field_name for m in state.pending_questions[0].missing_info] if state else []

    parsed_args = await agent.followup_middleware.process_answer(
        session_id, answer, expected
    )

    # 合并已收集参数，继续执行
    merged = {**state.collected_info, **parsed_args}
    return await execute_with_params(agent, session_id, merged)
```

---

## 7. 配置项 `config/followup.yaml`

```yaml
follow_up:
  enabled: true                       # 是否启用主动追问
  max_follow_ups: 3                   # 单次请求最大追问次数
  max_questions_per_turn: 2           # 每次最多问几个问题
  timeout_seconds: 300                # 等待用户回答的超时时间（秒）

  detection:
    use_llm: true                     # 使用 LLM 检测信息不足
    confidence_threshold: 0.7         # 低于此阈值时触发追问

  high_risk:
    auto_detect: true                 # 自动检测高风险操作
    require_confirmation: true        # 高风险操作必须确认
    patterns_file: "data/high_risk_patterns.json"  # 自定义风险模式

  generation:
    use_llm: true                     # 使用 LLM 生成追问内容
    tone: friendly                    # 语气风格：friendly | professional | casual
```

---

## 8. 与现有模块的交互

| 模块 | 交互方式 |
|------|---------|
| `06-agent-core` | 在意图识别后、工具执行前插入追问检查 |
| `04-session-manager` | 追问状态存储在 session 上下文中 |
| `03-tool-system` | 从工具注册表获取参数定义，判断必填项 |
| `17-prompt-template-management` | 追问 prompt 使用模板引擎管理 |

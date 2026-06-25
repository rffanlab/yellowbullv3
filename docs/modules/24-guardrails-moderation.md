# 安全护栏与内容审核详细设计（Guardrails & Moderation）

## 1. 职责边界

| 职责 | 说明 |
|------|------|
| **输入审核** | 检测用户输入中的违规、恶意、敏感内容 |
| **输出审核** | 过滤 LLM 回复中的不当内容 |
| **提示词注入防护** | 检测和防御 prompt injection / jailbreak 攻击 |
| **PII 保护** | 识别和脱敏个人身份信息 |
| **合规检查** | 确保交互符合法律法规要求 |

---

## 2. 协议设计 `guardrails/protocol.py`

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ViolationType(str, Enum):
    # 内容安全类
    HATE_SPEECH = "hate_speech"           # 仇恨言论
    HARASSMENT = "harassment"             # 骚扰
    SEXUAL_CONTENT = "sexual_content"     # 色情内容
    VIOLENCE = "violence"                 # 暴力内容
    SELF_HARM = "self_harm"               # 自残引导

    # 安全类
    PROMPT_INJECTION = "prompt_injection"  # 提示词注入
    JAILBREAK = "jailbreak"               # 越狱攻击
    MALICIOUS_CODE = "malicious_code"     # 恶意代码

    # 隐私类
    PII_EXPOSURE = "pii_exposure"         # 个人信息泄露

    # 合规类
    ILLEGAL_ACTIVITY = "illegal_activity"  # 违法活动引导
    MEDICAL_ADVICE = "medical_advice"      # 不当医疗建议
    FINANCIAL_ADVICE = "financial_advice"  # 不当金融建议


class SeverityLevel(str, Enum):
    LOW = "low"           # 低风险，记录日志即可
    MEDIUM = "medium"     # 中风险，需要警告用户
    HIGH = "high"         # 高风险，拦截并拒绝
    CRITICAL = "critical"  # 极高风险，拦截 + 上报


@dataclass(frozen=True)
class Violation:
    """违规内容描述"""
    violation_type: ViolationType   # 违规类型
    severity: SeverityLevel         # 严重程度
    description: str                # 人类可读的描述
    matched_text: str | None = None  # 匹配到的文本片段
    confidence: float = 0.0         # 检测置信度 [0, 1]


@dataclass(frozen=True)
class ModerationResult:
    """审核结果"""
    is_safe: bool                   # 内容是否安全
    violations: list[Violation] = field(default_factory=list)  # 违规列表
    score: float = 1.0              # 安全评分 [0, 1], 1 最安全
    action: str = "allow"           # 处理动作：allow | warn | block | escalate


@dataclass(frozen=True)
class GuardrailPolicy:
    """护栏策略配置"""
    name: str                       # 策略名称
    enabled: bool = True            # 是否启用
    blocked_types: list[ViolationType] = field(default_factory=list)  # 拦截的违规类型
    warn_types: list[ViolationType] = field(default_factory=list)     # 警告的违规类型
    min_confidence: float = 0.7     # 最低置信度阈值
    custom_instructions: str = ""   # 自定义审核指令


@dataclass(frozen=True)
class PIIInfo:
    """个人信息"""
    pii_type: str                   # 类型：phone | email | id_card | name | address
    raw_value: str                  # 原始值
    masked_value: str               # 脱敏后的值
```

---

## 3. 内容审核器 `guardrails/moderator.py`

```python
"""
内容审核模块。

支持多级审核策略：
1. 关键词匹配（快速，<1ms）
2. 正则表达式检测（中等精度）
3. LLM 语义分析（高精度，用于边界情况）
"""

import logging
import re
from typing import Any, Optional

from guardrails.protocol import (
    ModerationResult,
    SeverityLevel,
    Violation,
    ViolationType,
)

logger = logging.getLogger(__name__)


class KeywordModerator:
    """
    关键词审核器。

    基于预定义词库进行快速匹配，作为第一道防线。
    """

    # 内置敏感词分类（实际使用应从配置文件加载）
    DEFAULT_KEYWORDS: dict[ViolationType, list[str]] = {
        ViolationType.HATE_SPEECH: [],       # 仇恨言论关键词
        ViolationType.VIOLENCE: [],          # 暴力相关关键词
        ViolationType.SELF_HARM: [],         # 自残相关关键词
    }

    def __init__(self, keywords: dict[ViolationType, list[str]] | None = None):
        self._keywords: dict[ViolationType, set[str]] = {}
        for vtype, words in (keywords or self.DEFAULT_KEYWORDS).items():
            self._keywords[vtype] = {w.lower() for w in words}

    def check(self, text: str) -> list[Violation]:
        """检查文本是否包含敏感关键词"""
        violations = []
        text_lower = text.lower()

        for vtype, keywords in self._keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    violations.append(Violation(
                        violation_type=vtype,
                        severity=self._get_severity(vtype),
                        description=f"检测到敏感内容：{vtype.value}",
                        matched_text=keyword,
                        confidence=0.8,
                    ))

        return violations

    def _get_severity(self, vtype: ViolationType) -> SeverityLevel:
        """根据违规类型确定严重程度"""
        critical_types = {ViolationType.SELF_HARM, ViolationType.JAILBREAK}
        high_types = {ViolationType.HATE_SPEECH, ViolationType.VIOLENCE, ViolationType.PROMPT_INJECTION}

        if vtype in critical_types:
            return SeverityLevel.CRITICAL
        if vtype in high_types:
            return SeverityLevel.HIGH
        return SeverityLevel.MEDIUM


class RegexModerator:
    """
    正则表达式审核器。

    用于检测特定模式的内容，如：
    - PII（手机号、身份证号等）
    - 恶意代码模式
    - Prompt injection 常见模式
    """

    PATTERNS = {
        ViolationType.PII_EXPOSURE: [
            # 中国大陆手机号
            (r"1[3-9]\d{9}", "phone"),
            # 中国大陆身份证号（18位）
            (r"\b\d{17}[\dxX]\b", "id_card"),
            # 邮箱地址
            (r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "email"),
        ],
        ViolationType.PROMPT_INJECTION: [
            # 常见 prompt injection 模式
            (r"(?i)(ignore| disregard)\s+(all\s+)?previous\s+(instructions|commands)", "injection"),
            (r"(?i)system\s*:\s*(ignore|override|bypass)", "injection"),
            (r"(?i)(act as|pretend to be)\s+a(n?)?\s+(developer|admin|system)", "jailbreak"),
        ],
    }

    def check(self, text: str) -> list[Violation]:
        """使用正则表达式检测违规模式"""
        violations = []

        for vtype, patterns in self.PATTERNS.items():
            for pattern, label in patterns:
                matches = re.finditer(pattern, text)
                for match in matches:
                    violations.append(Violation(
                        violation_type=vtype,
                        severity=self._get_severity(vtype),
                        description=f"检测到模式：{label}",
                        matched_text=match.group(),
                        confidence=0.75,
                    ))

        return violations

    def _get_severity(self, vtype: ViolationType) -> SeverityLevel:
        if vtype == ViolationType.PROMPT_INJECTION:
            return SeverityLevel.HIGH
        if vtype == ViolationType.JAILBREAK:
            return SeverityLevel.CRITICAL
        if vtype == ViolationType.PII_EXPOSURE:
            return SeverityLevel.MEDIUM
        return SeverityLevel.LOW


class LLMModerator:
    """
    基于 LLM 的语义审核器。

    用于检测关键词和正则无法覆盖的边界情况，如：
    - 隐式仇恨言论（表面正常但隐含歧视）
    - 复杂的 jailbreak 攻击
    - 上下文相关的违规内容
    """

    MODERATION_PROMPT = """你是一个内容审核助手。请分析以下文本是否存在安全问题。

待审核文本：
{text}

请以 JSON 格式返回审核结果：
{{
  "is_safe": true/false,
  "violations": [
    {{
      "type": "违规类型",
      "severity": "low" | "medium" | "high" | "critical",
      "description": "说明",
      "confidence": 0.95
    }}
  ],
  "score": 0.8,           // 安全评分 [0, 1]
  "reasoning": "简要分析理由"
}}

违规类型可选值：hate_speech, harassment, sexual_content, violence, self_harm,
prompt_injection, jailbreak, malicious_code, pii_exposure, illegal_activity"""

    def __init__(self, llm_client: Any):
        self._llm = llm_client

    async def check(self, text: str) -> ModerationResult:
        """使用 LLM 进行语义审核"""
        prompt = self.MODERATION_PROMPT.format(text=text[:4000])

        try:
            import json as _json

            response = await self._llm.generate(prompt)
            result = _json.loads(response.strip())

            violations = []
            for v in result.get("violations", []):
                violations.append(Violation(
                    violation_type=ViolationType(v["type"]),
                    severity=SeverityLevel(v["severity"]),
                    description=v["description"],
                    confidence=v.get("confidence", 0.5),
                ))

            return ModerationResult(
                is_safe=result.get("is_safe", True),
                violations=violations,
                score=result.get("score", 1.0),
            )

        except Exception as e:
            logger.error(f"LLM moderation failed: {e}")
            # Fallback：假设安全（避免误拦截）
            return ModerationResult(is_safe=True)


class ContentModerator:
    """
    综合内容审核器。

    串联多级审核策略，平衡性能和精度。
    """

    def __init__(
        self,
        llm_client: Any | None = None,
        use_llm_fallback: bool = True,
    ):
        self._keyword = KeywordModerator()
        self._regex = RegexModerator()
        self._llm = LLMModerator(llm_client) if llm_client else None
        self._use_llm = use_llm_fallback

    async def moderate(self, text: str) -> ModerationResult:
        """
        执行多级内容审核。

        流程：
        1. 关键词匹配 → 发现严重违规直接拦截
        2. 正则检测 → 补充模式匹配
        3. LLM 语义分析 → 处理边界情况（仅当前两步不确定时）
        """
        # Level 1: 关键词
        keyword_violations = self._keyword.check(text)
        critical = [v for v in keyword_violations if v.severity == SeverityLevel.CRITICAL]

        if critical:
            return ModerationResult(
                is_safe=False,
                violations=critical,
                score=0.0,
                action="block",
            )

        # Level 2: 正则
        regex_violations = self._regex.check(text)
        all_violations = keyword_violations + regex_violations

        high_severity = [v for v in all_violations if v.severity in (SeverityLevel.HIGH, SeverityLevel.CRITICAL)]

        if high_severity:
            return ModerationResult(
                is_safe=False,
                violations=high_severity,
                score=0.2,
                action="block",
            )

        # Level 3: LLM（仅当前两步没有明确结论时）
        if self._llm and self._use_llm and not all_violations:
            llm_result = await self._llm.check(text)
            if not llm_result.is_safe:
                return ModerationResult(
                    is_safe=False,
                    violations=llm_result.violations,
                    score=llm_result.score,
                    action="block" if any(v.severity == SeverityLevel.CRITICAL for v in llm_result.violations) else "warn",
                )

        # 安全或低风险
        return ModerationResult(
            is_safe=True,
            violations=all_violations,
            score=1.0 - len(all_violations) * 0.1,
            action="allow" if not all_violations else "warn",
        )

    def add_keywords(self, violation_type: ViolationType, keywords: list[str]) -> None:
        """动态添加敏感关键词"""
        existing = self._keyword._keywords.get(violation_type, set())
        self._keyword._keywords[violation_type] = existing | {k.lower() for k in keywords}
```

---

## 4. PII 脱敏器 `guardrails/pii_masker.py`

```python
"""
个人身份信息（PII）检测与脱敏。

支持：
- 手机号、身份证号、邮箱等常见 PII 类型
- 自定义正则模式
- 多种脱敏策略（替换、掩码、哈希）
"""

import hashlib
import logging
import re
from typing import Any

from guardrails.protocol import PIIInfo

logger = logging.getLogger(__name__)


class PIIMasker:
    """PII 检测与脱敏器"""

    DEFAULT_PATTERNS = [
        ("phone", r"1[3-9]\d{9}", lambda m: f"{m[:3]}****{m[-4:]}"),
        ("id_card", r"\b\d{17}[\dxX]\b", lambda m: f"{m[:6]}**********{m[-2:]}"),
        ("email", r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", lambda m: f"{m[0]:3}***@***.com"),
        ("bank_card", r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b", lambda m: f"**** **** **** {m[-4:]}"),
    ]

    def __init__(self, custom_patterns: list[tuple[str, str, Any]] | None = None):
        self._patterns = [
            (name, re.compile(pattern), masker)
            for name, pattern, masker in self.DEFAULT_PATTERNS
        ]
        if custom_patterns:
            self._patterns.extend(
                (name, re.compile(pattern), masker)
                for name, pattern, masker in custom_patterns
            )

    def detect(self, text: str) -> list[PIIInfo]:
        """检测文本中的 PII"""
        found = []
        for name, regex, _ in self._patterns:
            for match in regex.finditer(text):
                raw = match.group()
                found.append(PIIInfo(
                    pii_type=name,
                    raw_value=raw,
                    masked_value=self._mask(raw, name),
                ))
        return found

    def mask(self, text: str) -> tuple[str, list[PIIInfo]]:
        """
        脱敏文本中的 PII。

        Returns:
            (脱敏后的文本, 检测到的 PII 列表)
        """
        pii_list = []

        for name, regex, masker in self._patterns:
            def replacer(match):
                raw = match.group()
                masked = masker(raw)
                pii_list.append(PIIInfo(pii_type=name, raw_value=raw, masked_value=masked))
                return masked

            text = regex.sub(replacer, text)

        return text, pii_list

    def _mask(self, value: str, pii_type: str) -> str:
        """根据类型脱敏"""
        for _, _, masker in self._patterns:
            if pii_type == _:
                try:
                    return masker(value)
                except Exception:
                    pass
        # 默认哈希脱敏
        return hashlib.sha256(value.encode()).hexdigest()[:8]
```

---

## 5. Guardrails 中间件 `guardrails/middleware.py`

```python
"""
Guardrails 中间件。

挂载在 Agent 消息处理管道的最外层，作为安全网关：
- 用户输入 → 审核 → （通过）→ Agent 处理
- LLM 输出 → 审核 → （通过）→ 返回用户
"""

import logging
from typing import Any, Optional

from guardrails.moderator import ContentModerator
from guardrails.pii_masker import PIIMasker
from guardrails.protocol import (
    GuardrailPolicy,
    ModerationResult,
    SeverityLevel,
    ViolationType,
)

logger = logging.getLogger(__name__)


class GuardrailsMiddleware:
    """
    安全护栏中间件。

    处理流程：
    ┌──────────┐   审核     ┌───────────┐   审核     ┌──────────┐
    │ User Input│──────────▶│ Agent Core │──────────▶│ Response │
    └──────────┘            └───────────┘            └──────────┘
         │                      │                       │
         ▼                      ▼                       ▼
      拦截/警告              PII脱敏               过滤不当内容
    """

    def __init__(
        self,
        moderator: ContentModerator,
        pii_masker: PIIMasker | None = None,
        policy: GuardrailPolicy | None = None,
    ):
        self._moderator = moderator
        self._pii_masker = pii_masker or PIIMasker()
        self._policy = policy or GuardrailPolicy(name="default")

    async def check_input(self, text: str) -> tuple[bool, str | None]:
        """
        审核用户输入。

        Returns:
            (是否通过, 拒绝原因或 None)
        """
        if not self._policy.enabled:
            return True, None

        result = await self._moderator.moderate(text)

        if result.action == "block":
            violations_desc = "; ".join(
                f"{v.violation_type.value}({v.severity.value})" for v in result.violations
            )
            logger.warning(f"Input blocked: {violations_desc}")
            return False, "您的输入包含不当内容，请修改后重试。"

        if result.action == "warn":
            logger.info(f"Input warning: {result.violations}")

        # PII 脱敏（可选）
        if self._pii_masker:
            masked_text, pii_info = self._pii_masker.mask(text)
            if pii_info:
                logger.info(f"Masked {len(pii_info)} PII items in input")
                text = masked_text

        return True, None

    async def check_output(self, text: str) -> tuple[bool, str]:
        """
        审核 LLM 输出。

        Returns:
            (是否通过, 最终返回文本)
        """
        if not self._policy.enabled:
            return True, text

        result = await self._moderator.moderate(text)

        if result.action == "block":
            logger.warning(f"Output blocked: {result.violations}")
            return False, "抱歉，我无法提供此内容的回复。"

        # PII 脱敏输出
        if self._pii_masker:
            masked_text, _ = self._pii_masker.mask(text)
            return True, masked_text

        return True, text

    def update_policy(self, policy: GuardrailPolicy) -> None:
        """更新护栏策略"""
        self._policy = policy
```

---

## 6. Agent 核心集成示例

```python
# 在 agent 主循环中：
async def process_message(agent, session_id: str, user_input: str):
    # === 输入审核 ===
    is_safe, reject_reason = await agent.guardrails.check_input(user_input)
    if not is_safe:
        return await agent.respond(session_id, reject_reason or "内容不安全")

    # ... Agent 正常处理 ...

    response_text = await agent.generate_response(session_id, user_input)

    # === 输出审核 ===
    is_safe, final_text = await agent.guardrails.check_output(response_text)
    if not is_safe:
        return await agent.respond(session_id, "抱歉，我无法提供此内容的回复。")

    return await agent.respond(session_id, final_text)
```

---

## 7. 配置项 `config/guardrails.yaml`

```yaml
guardrails:
  enabled: true                       # 是否启用安全护栏

  moderation:
    use_llm_fallback: true            # 关键词/正则未命中时使用 LLM 审核
    llm_model: "gpt-4o-mini"          # 审核使用的模型（低成本）
    min_confidence: 0.7               # 最低置信度阈值

  pii_protection:
    enabled: true                     # 是否启用 PII 脱敏
    mask_on_input: false              # 输入端脱敏（默认不脱，保留原始数据）
    mask_on_output: true              # 输出端脱敏

  policy:
    blocked_types:                    # 直接拦截的违规类型
      - jailbreak
      - prompt_injection
      - self_harm
    warn_types:                       # 仅警告的违规类型
      - pii_exposure
      - medical_advice
```

---

## 8. 与现有模块的交互

| 模块 | 交互方式 |
|------|---------|
| `06-agent-core` | GuardrailsMiddleware 挂载在消息管道最外层 |
| `10-security-governance` | 审核日志上报到安全治理模块 |
| `04-session-manager` | 违规记录关联 session，支持会话级封禁 |

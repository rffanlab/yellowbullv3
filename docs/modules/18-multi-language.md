# 多语言支持详细设计（Multi-language Support）

## 1. 职责边界

| 职责 | 说明 |
|------|------|
| **语言检测** | 自动识别用户输入的语言类型，支持中英文混合文本 |
| **翻译管道** | 在需要时进行跨语言翻译，保持语义一致性 |
| **多语言 Prompt** | 根据用户语言动态选择/生成对应语言的 system prompt |
| **响应语言控制** | 确保 Agent 回复与用户输入语言一致（或按配置切换） |
| **术语管理** | 维护领域术语的多语言映射表，保证专业词汇翻译准确 |

---

## 2. 协议设计 `i18n/protocol.py`

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Language(str, Enum):
    ZH_CN = "zh-CN"       # 简体中文
    ZH_TW = "zh-TW"       # 繁体中文
    EN_US = "en-US"       # 美式英语
    JA_JP = "ja-JP"       # 日语
    KO_KR = "ko-KR"       # 韩语


@dataclass(frozen=True)
class LanguageDetectionResult:
    """语言检测结果"""
    primary_language: Language              # 主要语言
    confidence: float                       # 置信度 [0, 1]
    languages: list[tuple[Language, float]] # 所有检测到的语言及占比


@dataclass(frozen=True)
class TranslationResult:
    """翻译结果"""
    source_text: str                        # 原文
    target_text: str                        # 译文
    source_lang: Language                   # 源语言
    target_lang: Language                   # 目标语言
    confidence: float | None = None         # 翻译置信度


@dataclass(frozen=True)
class I18nContext:
    """多语言上下文"""
    user_language: Language                 # 用户偏好语言
    detected_language: Language             # 当前输入检测到的语言
    response_language: Language             # 回复应使用的语言
    mixed_input: bool = False               # 是否为混合语言输入
    terminology_overrides: dict[str, str] = field(default_factory=dict)  # 术语替换表
```

---

## 3. 语言检测器 `i18n/detector.py`

```python
"""
多语言检测模块。

策略：
- 短文本 (<50 chars): 基于字符集统计 + 关键词匹配
- 长文本 (>=50 chars): LLM 辅助检测，提高混合语言的识别精度
- 缓存检测结果，避免重复计算
"""

import logging
from collections import Counter
from typing import Optional

from i18n.protocol import Language, LanguageDetectionResult

logger = logging.getLogger(__name__)


# CJK 字符范围
CJK_UNICODE_RANGES = [
    (0x4E00, 0x9FFF),   # CJK Unified Ideographs
    (0x3400, 0x4DBF),   # CJK Extension A
    (0x20000, 0x2A6DF), # CJK Extension B
    (0xFF00, 0xFFEF),   # Fullwidth ASCII
]

# 各语言特征词（用于辅助判断）
LANGUAGE_KEYWORDS: dict[Language, list[str]] = {
    Language.ZH_CN: ["的", "了", "是", "在", "我", "有", "不", "你", "他", "她"],
    Language.EN_US: ["the", "is", "are", "was", "were", "have", "has", "had", "will", "would"],
    Language.JA_JP: ["です", "ます", "ない", "ある", "する", "こと", "ため", "ので"],
    Language.KO_KR: ["합니다", "입니다", "하지", "있다", "하다", "것", "때문"],
}


def _count_cjk_chars(text: str) -> int:
    """统计 CJK 字符数量"""
    count = 0
    for char in text:
        cp = ord(char)
        if any(start <= cp <= end for start, end in CJK_UNICODE_RANGES):
            count += 1
    return count


def _detect_by_charset(text: str) -> LanguageDetectionResult:
    """基于字符集的轻量级检测"""
    cleaned = text.strip()
    if not cleaned:
        return LanguageDetectionResult(
            primary_language=Language.ZH_CN,
            confidence=0.5,
            languages=[(Language.ZH_CN, 0.5)],
        )

    total = len(cleaned)
    cjk_count = _count_cjk_chars(cleaned)
    cjk_ratio = cjk_count / total if total > 0 else 0

    # 统计各语言特征词出现次数
    lower_text = cleaned.lower()
    keyword_scores: dict[Language, float] = {}
    for lang, keywords in LANGUAGE_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in lower_text)
        if score > 0:
            keyword_scores[lang] = score

    # 综合判断
    if cjk_ratio > 0.3:
        # CJK 字符占比较高，可能是中文/日文/韩文
        if Language.JA_JP in keyword_scores or any(
            '\u3042' <= ord(c) <= '\u309f' or '\u30a0' <= ord(c) <= '\u30ff'
            for c in cleaned
        ):
            primary = Language.JA_JP if Language.JA_JP in keyword_scores else Language.ZH_CN
            confidence = min(0.95, 0.6 + cjk_ratio * 0.3)
        elif Language.KO_KR in keyword_scores or any(
            '\uac00' <= ord(c) <= '\ud7af' for c in cleaned
        ):
            primary = Language.KO_KR
            confidence = min(0.95, 0.6 + cjk_ratio * 0.3)
        else:
            primary = Language.ZH_CN
            confidence = min(0.95, 0.6 + cjk_ratio * 0.3)
    elif keyword_scores.get(Language.EN_US, 0) > 0 or cjk_ratio < 0.1:
        primary = Language.EN_US
        confidence = max(0.7, 1.0 - cjk_ratio)
    else:
        # 无法确定，默认中文
        primary = Language.ZH_CN
        confidence = 0.5

    return LanguageDetectionResult(
        primary_language=primary,
        confidence=confidence,
        languages=[(primary, confidence)],
    )


class LanguageDetector:
    """
    多语言检测器。

    使用两级策略：
    1. 快速路径：字符集统计 + 关键词匹配（<1ms）
    2. 精确路径：LLM 辅助检测（用于混合语言或低置信度情况）
    """

    def __init__(self, llm_client: Optional[Any] = None):
        self._llm = llm_client
        self._cache: dict[str, LanguageDetectionResult] = {}
        self._max_cache_size = 1000

    async def detect(
        self,
        text: str,
        use_llm: bool = False,
    ) -> LanguageDetectionResult:
        """
        检测文本语言。

        Args:
            text:     待检测的文本
            use_llm:  是否使用 LLM 进行精确检测（默认仅字符集检测）

        Returns:
            LanguageDetectionResult
        """
        cache_key = text[:200]  # 长文本截断作为缓存键
        if cache_key in self._cache:
            return self._cache[cache_key]

        result = _detect_by_charset(text)

        # 低置信度或强制使用 LLM 时，走精确路径
        if (result.confidence < 0.7 or use_llm) and self._llm:
            result = await self._detect_with_llm(text)

        # 缓存结果
        if len(self._cache) < self._max_cache_size:
            self._cache[cache_key] = result

        return result

    async def _detect_with_llm(self, text: str) -> LanguageDetectionResult:
        """使用 LLM 进行精确语言检测"""
        prompt = f"""分析以下文本的语言构成，返回 JSON：
{{"primary": "主要语言代码", "confidence": 0.95, "languages": [["lang_code", ratio]]}}

文本：{text[:1000]}"""

        # TODO: 调用 LLM 解析结果
        return _detect_by_charset(text)

    def is_mixed_language(self, text: str, threshold: float = 0.3) -> bool:
        """
        判断是否为混合语言文本。

        Args:
            text:      待检测文本
            threshold: 次要语言的最低占比阈值

        Returns:
            True 如果检测到多种语言且每种占比超过阈值
        """
        cjk_ratio = _count_cjk_chars(text) / max(len(text), 1)
        ascii_ratio = sum(1 for c in text if c.isascii()) / max(len(text), 1)

        # CJK 和 ASCII 都占一定比例，可能是混合语言
        return cjk_ratio > threshold and ascii_ratio > threshold
```

---

## 4. 翻译管道 `i18n/translator.py`

```python
"""
翻译管道。

支持两种翻译模式：
- LLM 翻译：高质量，适合短文本和关键内容（如 prompt、回复）
- API 翻译：低成本高吞吐，适合大批量文档处理
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any

from i18n.protocol import Language, TranslationResult

logger = logging.getLogger(__name__)


class BaseTranslator(ABC):
    """翻译器基类"""

    @property
    @abstractmethod
    def translator_name(self) -> str:
        ...

    @abstractmethod
    async def translate(
        self,
        text: str,
        source_lang: Language | None,
        target_lang: Language,
        context: str | None = None,
    ) -> TranslationResult:
        """
        翻译文本。

        Args:
            text:        待翻译文本
            source_lang: 源语言（None 则自动检测）
            target_lang: 目标语言
            context:     上下文信息，用于提高翻译质量
        """
        ...


class LLMTranslator(BaseTranslator):
    """
    基于 LLM 的翻译器。

    优势：语境感知、术语一致性、风格匹配
    劣势：成本高、延迟高
    """

    def __init__(self, llm_client: Any, model: str = "gpt-4o-mini"):
        self._llm = llm_client
        self._model = model

    @property
    def translator_name(self) -> str:
        return f"llm:{self._model}"

    async def translate(
        self,
        text: str,
        source_lang: Language | None,
        target_lang: Language,
        context: str | None = None,
    ) -> TranslationResult:
        if not text.strip():
            return TranslationResult(
                source_text=text,
                target_text=text,
                source_lang=source_lang or Language.ZH_CN,
                target_lang=target_lang,
            )

        src_label = source_lang.value if source_lang else "auto-detect"
        ctx_hint = f"\n上下文：{context}" if context else ""

        prompt = f"""请将以下文本从 {src_label} 翻译为 {target_lang.value}。
要求：保持原意、语气一致，专业术语准确。只返回翻译结果，不要解释。

原文：{text}{ctx_hint}

翻译："""

        # TODO: 调用 LLM 获取翻译结果
        translated = text  # placeholder

        return TranslationResult(
            source_text=text,
            target_text=translated.strip(),
            source_lang=source_lang or Language.ZH_CN,
            target_lang=target_lang,
        )


class TerminologyManager:
    """
    术语管理器。

    维护领域术语的多语言映射，确保关键概念翻译一致。
    """

    def __init__(self):
        self._terms: dict[Language, dict[str, str]] = {}

    def load_terms(self, filepath: str) -> None:
        """从 JSON/YAML 文件加载术语表"""
        # TODO: 实现文件加载逻辑
        logger.info(f"Loaded terminology from {filepath}")

    def add_term(
        self,
        source_lang: Language,
        target_lang: Language,
        original: str,
        translation: str,
    ) -> None:
        """添加术语映射"""
        if target_lang not in self._terms:
            self._terms[target_lang] = {}
        self._terms[target_lang][original.lower()] = translation

    def apply(
        self,
        text: str,
        target_lang: Language,
    ) -> str:
        """将术语映射应用到翻译结果"""
        if target_lang not in self._terms:
            return text

        for original, translation in self._terms[target_lang].items():
            text = text.replace(original, translation)
        return text


class TranslationPipeline:
    """
    翻译管道编排。

    流程：检测源语言 → 选择翻译器 → 执行翻译 → 术语校正 → 缓存结果
    """

    def __init__(
        self,
        translators: list[BaseTranslator],
        terminology: TerminologyManager | None = None,
    ):
        self._translators = translators
        self._primary = translators[0] if translators else None
        self._terminology = terminology
        self._cache: dict[str, TranslationResult] = {}

    async def translate(
        self,
        text: str,
        target_lang: Language,
        source_lang: Language | None = None,
        context: str | None = None,
    ) -> TranslationResult:
        if not self._primary:
            raise RuntimeError("No translator available")

        cache_key = f"{text[:100]}|{source_lang}|{target_lang}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        result = await self._primary.translate(
            text, source_lang, target_lang, context
        )

        # 术语校正
        if self._terminology:
            result = TranslationResult(
                **result.__dict__,
                target_text=self._terminology.apply(result.target_text, target_lang),
            )

        if len(self._cache) < 5000:
            self._cache[cache_key] = result

        return result
```

---

## 5. Prompt 多语言适配 `i18n/prompt_adapter.py`

```python
"""
Prompt 多语言适配器。

根据用户语言动态调整 system prompt，确保 Agent 行为一致但表达自然。
"""

import logging
from typing import Any

from i18n.protocol import I18nContext, Language

logger = logging.getLogger(__name__)


class PromptAdapter:
    """
    System Prompt 多语言适配器。

    策略：
    - 维护每种语言的 system prompt 模板
    - 支持 fallback 链（如 zh-TW → zh-CN → en-US）
    - 动态注入语言相关的行为指令
    """

    DEFAULT_PROMPTS: dict[Language, str] = {
        Language.ZH_CN: (
            "你是一个专业的 AI 助手。请用中文回复用户，"
            "保持专业、友好、简洁的风格。"
        ),
        Language.EN_US: (
            "You are a professional AI assistant. Please respond in English, "
            "maintaining a professional, friendly, and concise style."
        ),
        Language.JA_JP: (
            "あなたはプロフェッショナルなAIアシスタントです。"
            "日本語で返信してください。"
        ),
    }

    def __init__(self):
        self._custom_prompts: dict[Language, str] = {}
        self._fallback_chain: dict[Language, list[Language]] = {
            Language.ZH_TW: [Language.ZH_CN, Language.EN_US],
            Language.KO_KR: [Language.JA_JP, Language.EN_US],
        }

    def register_prompt(self, language: Language, prompt: str) -> None:
        """注册自定义 prompt"""
        self._custom_prompts[language] = prompt

    def get_system_prompt(
        self,
        context: I18nContext,
        extra_instructions: list[str] | None = None,
    ) -> str:
        """
        获取适配用户语言的 system prompt。

        Args:
            context:           多语言上下文
            extra_instructions: 额外指令列表

        Returns:
            完整的 system prompt
        """
        lang = context.response_language
        prompt = self._custom_prompts.get(lang) or self._resolve_prompt(lang)

        # 混合输入时添加特殊处理指令
        if context.mixed_input:
            prompt += (
                f"\n\n注意：用户可能使用混合语言输入。"
                f"请理解混合内容，并用 {lang.value} 回复。"
            )

        # 注入额外指令
        if extra_instructions:
            prompt += "\n\n额外要求：" + "\n".join(f"- {i}" for i in extra_instructions)

        return prompt

    def _resolve_prompt(self, language: Language) -> str:
        """解析 prompt，支持 fallback 链"""
        if language in self._custom_prompts:
            return self._custom_prompts[language]
        if language in self.DEFAULT_PROMPTS:
            return self.DEFAULT_PROMPTS[language]

        # Fallback chain
        for fallback in self._fallback_chain.get(language, []):
            if fallback in self._custom_prompts:
                return self._custom_prompts[fallback]
            if fallback in self.DEFAULT_PROMPTS:
                return self.DEFAULT_PROMPTS[fallback]

        # 最终 fallback 到英文
        logger.warning(f"No prompt for {language}, falling back to en-US")
        return self.DEFAULT_PROMPTS[Language.EN_US]

    def adapt_response_language(
        self,
        detected_lang: Language,
        user_preference: Language | None = None,
    ) -> Language:
        """
        确定回复应使用的语言。

        优先级：用户偏好 > 检测到的语言 > 默认中文
        """
        if user_preference:
            return user_preference
        return detected_lang or Language.ZH_CN
```

---

## 6. 集成到 Agent 核心 `i18n/integration.py`

```python
"""
多语言模块与 Agent 核心的集成。

在消息处理管道的入口处进行语言检测，
在 prompt 构建阶段注入语言适配指令。
"""

import logging

from i18n.detector import LanguageDetector
from i18n.protocol import I18nContext, Language
from i18n.prompt_adapter import PromptAdapter
from i18n.translator import TranslationPipeline, TerminologyManager

logger = logging.getLogger(__name__)


class I18nMiddleware:
    """
    多语言中间件。

    挂载在 Agent 消息处理管道中，负责：
    1. 检测用户输入语言
    2. 构建 I18nContext
    3. 适配 system prompt
    4. （可选）翻译混合语言输入为统一语言供 LLM 处理
    """

    def __init__(
        self,
        detector: LanguageDetector,
        prompt_adapter: PromptAdapter,
        translator: TranslationPipeline | None = None,
        default_language: Language = Language.ZH_CN,
    ):
        self._detector = detector
        self._prompt_adapter = prompt_adapter
        self._translator = translator
        self._default_language = default_language

    async def process_input(
        self,
        user_message: str,
        user_preference: Language | None = None,
    ) -> I18nContext:
        """
        处理用户输入，返回多语言上下文。

        Args:
            user_message:    用户原始消息
            user_preference: 用户偏好语言（来自 profile/session）

        Returns:
            I18nContext
        """
        detection = await self._detector.detect(user_message)
        mixed = self._detector.is_mixed_language(user_message)

        response_lang = self._prompt_adapter.adapt_response_language(
            detection.primary_language, user_preference
        )

        return I18nContext(
            user_language=user_preference or self._default_language,
            detected_language=detection.primary_language,
            response_language=response_lang,
            mixed_input=mixed,
        )

    def build_system_prompt(
        self,
        context: I18nContext,
        base_prompt: str,
        extra_instructions: list[str] | None = None,
    ) -> str:
        """构建多语言适配的 system prompt"""
        adapted = self._prompt_adapter.get_system_prompt(context, extra_instructions)

        # 将基础 prompt 与语言适配指令合并
        if context.detected_language != Language.EN_US and not context.mixed_input:
            return adapted + "\n\n" + base_prompt
        else:
            # 混合输入或英文时，可能需要翻译 base prompt
            return adapted + "\n\n" + base_prompt

    async def translate_for_processing(
        self,
        text: str,
        context: I18nContext,
        target_lang: Language = Language.ZH_CN,
    ) -> str:
        """
        将混合语言输入翻译为统一语言，供内部处理。

        注意：仅在混合语言且 LLM 难以直接处理时使用。
        大多数现代 LLM 可以直接处理多语言输入。
        """
        if not self._translator or not context.mixed_input:
            return text

        result = await self._translator.translate(
            text, target_lang=target_lang, source_lang=context.detected_language
        )
        return result.target_text


def create_i18n_middleware(llm_client=None) -> I18nMiddleware:
    """创建多语言中间件的工厂函数"""
    detector = LanguageDetector(llm_client=llm_client)
    prompt_adapter = PromptAdapter()
    terminology = TerminologyManager()

    # 默认不启用翻译管道（现代 LLM 原生支持多语言）
    translator = None

    return I18nMiddleware(
        detector=detector,
        prompt_adapter=prompt_adapter,
        translator=translator,
    )
```

---

## 7. 配置项 `config/i18n.yaml`

```yaml
i18n:
  default_language: zh-CN           # 默认语言
  supported_languages:              # 支持的语言列表
    - zh-CN
    - en-US
    - ja-JP
    - ko-KR
    - zh-TW

  detection:
    use_llm_for_low_confidence: true   # 低置信度时使用 LLM 精确检测
    confidence_threshold: 0.7           # 触发精确检测的阈值
    cache_enabled: true                 # 启用检测结果缓存
    max_cache_size: 1000

  translation:
    enabled: false                      # 默认不启用翻译管道
    provider: llm                       # llm | api
    model: gpt-4o-mini                  # LLM 翻译使用的模型
    cache_enabled: true                 # 启用翻译结果缓存

  terminology:
    file_path: "data/terminology.json"  # 术语表文件路径
    auto_apply: true                    # 自动应用到翻译结果

  prompt:
    fallback_chain:                     # Prompt fallback 顺序
      zh-TW: [zh-CN, en-US]
      ko-KR: [ja-JP, en-US]
```

---

## 8. 数据流图

```
用户输入 (混合语言)
        │
        ▼
┌───────────────┐
│  Language     │  检测主要语言、判断是否混合
│  Detector     │
└───────┬───────┘
        │ I18nContext
        ▼
┌───────────────┐
│  Prompt       │  根据目标语言选择/适配 system prompt
│  Adapter      │
└───────┬───────┘
        │ adapted_system_prompt
        ▼
┌───────────────┐
│  LLM          │  现代 LLM 原生支持多语言，无需翻译中间层
│  Inference    │
└───────┬───────┘
        │ response (目标语言)
        ▼
┌───────────────┐     仅在需要时启用
│  Terminology  │→ 术语校正 → 返回用户
│  Manager      │
└───────────────┘
```

---

## 9. 关键设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| **翻译策略** | 默认不翻译，依赖 LLM 原生多语言能力 | GPT-4/Claude 等模型已具备优秀多语言理解能力，额外翻译增加延迟和成本 |
| **何时启用翻译管道** | 仅对混合语言且检测置信度低时 | 平衡质量和性能 |
| **语言检测精度** | 两级策略：字符集快速检测 + LLM 精确检测 | 90% 场景可在 <1ms 内完成，复杂情况才走 LLM |
| **术语管理** | 独立术语表 + 翻译后校正 | 保证专业词汇一致性，不影响通用翻译质量 |

---

## 10. 与现有模块的交互

| 模块 | 交互方式 |
|------|---------|
| `02-llm-abstraction` | I18nContext 注入到 LLM 调用上下文 |
| `04-session-manager` | 用户偏好语言存储在 session profile 中 |
| `17-prompt-template-management` | Prompt 模板支持多语言版本管理 |
| `09-multi-modal` | ASR 结果携带检测到的语言信息 |

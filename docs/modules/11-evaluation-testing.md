# 评估与测试详细设计

## 1. 设计目标

| 目标 | 说明 |
|------|------|
| **质量保障** | 多维度评估 Agent 回答质量，持续监控退化 |
| **自动化测试** | 单元测试、集成测试、端到端测试全覆盖 |
| **基准对比** | 支持多模型横向对比，量化改进效果 |
| **回归检测** | CI/CD 流水线中自动运行回归测试集 |

---

## 2. 协议设计 `evaluation/protocol.py`

```python
from dataclasses import dataclass, field
from enum import Enum


class MetricType(Enum):
    """评估指标类型"""
    BLEU = "bleu"              # n-gram 重叠度（机器翻译）
    ROUGE = "rouge"            # 召回导向的 n-gram（摘要）
    METEOR = "meteor"          # 语义对齐（同义词、词干）
    BERTSCORE = "bertscore"    # 基于 BERT 嵌入的语义相似度
    LLM_JUDGE = "llm_judge"   # LLM 作为裁判打分
    HALLUCINATION = "hallucination"  # 幻觉检测
    LATENCY = "latency"        # 响应延迟（P50/P95/P99）
    TOKEN_USAGE = "token_usage"     # Token 消耗统计


@dataclass(frozen=True)
class TestCase:
    """单条测试用例"""
    id: str                                    # 唯一 ID
    input: str                                 # 用户输入
    expected_output: str | None = None         # 期望输出（可为空，用于 LLM Judge）
    criteria: dict[str, float] | None = None   # 评分标准：{"relevance": 1.0, "accuracy": 1.0}
    tags: list[str] | None = None              # 标签分类


@dataclass(frozen=True)
class TestResult:
    """单条测试结果"""
    test_case_id: str                          # 测试用例 ID
    actual_output: str                         # Agent 实际输出
    metrics: dict[str, float]                  # 各指标得分
    passed: bool                               # 是否通过（综合判定）
    latency_ms: float = 0.0                    # 响应延迟


@dataclass(frozen=True)
class EvaluationReport:
    """评估报告"""
    test_suite_name: str                       # 测试集名称
    total_cases: int                           # 总用例数
    passed_cases: int                          # 通过用例数
    pass_rate: float                           # 通过率 [0, 1]
    average_metrics: dict[str, float]          # 各指标平均分
    latency_p50: float = 0.0                   # P50 延迟（ms）
    latency_p95: float = 0.0                   # P95 延迟（ms）
    latency_p99: float = 0.0                   # P99 延迟（ms）
    results: list[TestResult] | None = None    # 详细结果


@dataclass(frozen=True)
class ModelComparisonReport:
    """多模型对比报告"""
    models: list[str]                          # 参评模型列表
    metrics: dict[str, dict[str, float]]       # model_name → {metric_name: score}
    winner: dict[str, str] | None = None       # 各指标最优模型：{metric_name: model_name}
```

---

## 3. 测试用例管理 `evaluation/test_suite.py`

```python
import json
from pathlib import Path
from evaluation.protocol import TestCase


class TestSuite:
    """
    测试套件管理器。

    支持从 JSON/YAML 文件加载测试集，也支持动态构建。
    """

    def __init__(self, name: str):
        self._name = name
        self._cases: list[TestCase] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def cases(self) -> list[TestCase]:
        return self._cases

    def add_case(
        self,
        input_text: str,
        expected_output: str | None = None,
        criteria: dict[str, float] | None = None,
        tags: list[str] | None = None,
        case_id: str | None = None,
    ) -> TestCase:
        """添加测试用例"""
        import uuid
        tc = TestCase(
            id=case_id or str(uuid.uuid4()),
            input=input_text,
            expected_output=expected_output,
            criteria=criteria,
            tags=tags,
        )
        self._cases.append(tc)
        return tc

    @classmethod
    def from_json(cls, path: str | Path, name: str | None = None) -> "TestSuite":
        """从 JSON 文件加载测试集"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        suite_name = name or data.get("name", "unnamed_suite")
        suite = cls(suite_name)

        for case_data in data.get("cases", []):
            suite.add_case(
                input_text=case_data["input"],
                expected_output=case_data.get("expected_output"),
                criteria=case_data.get("criteria"),
                tags=case_data.get("tags"),
                case_id=case_data.get("id"),
            )

        return suite

    def to_json(self, path: str | Path):
        """导出测试集为 JSON"""
        data = {
            "name": self._name,
            "cases": [
                {
                    "id": tc.id,
                    "input": tc.input,
                    "expected_output": tc.expected_output,
                    "criteria": tc.criteria,
                    "tags": tc.tags,
                }
                for tc in self._cases
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def filter_by_tags(self, tags: list[str]) -> list[TestCase]:
        """按标签过滤测试用例"""
        tag_set = set(tags)
        return [tc for tc in self._cases if tc.tags and set(tc.tags) & tag_set]


# ==================== 内置测试集 ====================

def create_general_knowledge_suite() -> TestSuite:
    """通用知识问答测试集（示例）"""
    suite = TestSuite("general-knowledge")

    cases = [
        ("Python 中 list 和 tuple 的区别是什么？", None, {"accuracy": 1.0}, ["programming"]),
        ("光合作用的过程是怎样的？", None, {"accuracy": 1.0, "completeness": 1.0}, ["science"]),
        ("请解释什么是微服务架构？", None, {"accuracy": 1.0, "clarity": 1.0}, ["architecture"]),
        ("2024年奥运会在哪里举办？", "巴黎", {"accuracy": 1.0}, ["facts"]),
    ]

    for inp, expected, criteria, tags in cases:
        suite.add_case(inp, expected, criteria, tags)

    return suite


def create_tool_usage_suite() -> TestSuite:
    """工具调用测试集（示例）"""
    suite = TestSuite("tool-usage")

    cases = [
        ("帮我搜索一下最近的天气", None, {"tool_selection": 1.0}, ["web_search"]),
        ("计算 23456 × 78901 的结果", "1850663856", {"accuracy": 1.0, "tool_usage": 1.0}, ["calculator"]),
        ("帮我写一段 Python 代码实现快速排序", None, {"code_quality": 1.0}, ["coding"]),
    ]

    for inp, expected, criteria, tags in cases:
        suite.add_case(inp, expected, criteria, tags)

    return suite


def create_multilingual_suite() -> TestSuite:
    """多语言测试集"""
    suite = TestSuite("multilingual")

    cases = [
        ("请用中文解释量子计算", None, {"language": "zh"}, ["chinese"]),
        ("Explain blockchain in simple English", None, {"language": "en"}, ["english"]),
        ("量子計算を日本語で説明してください", None, {"language": "ja"}, ["japanese"]),
    ]

    for inp, expected, criteria, tags in cases:
        suite.add_case(inp, expected, criteria, tags)

    return suite
```

---

## 4. 评估指标 `evaluation/metrics.py`

```python
from evaluation.protocol import TestCase, TestResult


class BaseMetric:
    """评估指标基类"""

    @property
    def name(self) -> str:
        raise NotImplementedError

    async def score(
        self, test_case: TestCase, actual_output: str
    ) -> dict[str, float]:
        """计算指标得分，返回 {metric_name: score}"""
        raise NotImplementedError


class BleuMetric(BaseMetric):
    """BLEU 评分（n-gram 精确匹配）"""

    @property
    def name(self) -> str:
        return "bleu"

    async def score(
        self, test_case: TestCase, actual_output: str
    ) -> dict[str, float]:
        if not test_case.expected_output:
            return {"bleu": 0.0}

        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            reference = test_case.expected_output.split()
            hypothesis = actual_output.split()

            if not hypothesis or not reference:
                return {"bleu": 0.0}

            smoother = SmoothingFunction().method1
            bleu = sentence_bleu([reference], hypothesis, smoothing_function=smoother)
            return {"bleu": round(bleu, 4)}
        except ImportError:
            return {"bleu": -1.0}  # nltk not available


class RougeMetric(BaseMetric):
    """ROUGE 评分（召回导向）"""

    @property
    def name(self) -> str:
        return "rouge"

    async def score(
        self, test_case: TestCase, actual_output: str
    ) -> dict[str, float]:
        if not test_case.expected_output:
            return {"rouge": 0.0}

        try:
            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
            scores = scorer.score(test_case.expected_output, actual_output)

            return {
                "rouge1": round(scores["rouge1"].fmeasure, 4),
                "rouge2": round(scores["rouge2"].fmeasure, 4),
                "rougeL": round(scores["rougeL"].fmeasure, 4),
            }
        except ImportError:
            return {"rouge": -1.0}


class BertScoreMetric(BaseMetric):
    """BERTScore（基于语义嵌入的相似度）"""

    def __init__(self, model: str = "bert-base-chinese"):
        self._model_name = model

    @property
    def name(self) -> str:
        return "bertscore"

    async def score(
        self, test_case: TestCase, actual_output: str
    ) -> dict[str, float]:
        if not test_case.expected_output:
            return {"bertscore": 0.0}

        try:
            import asyncio
            from bert_score import score as bert_score_fn
            loop = asyncio.get_event_loop()

            precision, recall, f1 = await loop.run_in_executor(
                None,
                lambda: bert_score_fn(
                    [actual_output], [test_case.expected_output],
                    model_type=self._model_name, lang="zh"
                ),
            )
            return {"bertscore_f1": round(f1[0].item(), 4)}
        except ImportError:
            return {"bertscore": -1.0}


class LLMJudgeMetric(BaseMetric):
    """
    LLM-as-a-Judge：使用 LLM 对回答质量进行评分。

    这是最灵活、最接近人类判断的评估方式。
    """

    def __init__(self, llm_client, model: str = "gpt-4o"):
        self._client = llm_client
        self._model = model

    @property
    def name(self) -> str:
        return "llm_judge"

    async def score(
        self, test_case: TestCase, actual_output: str
    ) -> dict[str, float]:
        criteria = test_case.criteria or {
            "relevance": 1.0,     # 相关性
            "accuracy": 1.0,      # 准确性
            "clarity": 1.0,       # 清晰度
            "completeness": 1.0,  # 完整性
        }

        criteria_text = "\n".join(f"- {k}: 权重 {v}" for k, v in criteria.items())

        prompt = f"""你是一个评估专家。请对以下回答进行评分（每项 0-10 分）。

## 用户问题
{test_case.input}

## Agent 回答
{actual_output}

## 评分标准
{criteria_text}

请以 JSON 格式回复：
{{"relevance": <分数>, "accuracy": <分数>, "clarity": <分数>, "completeness": <分数>, "reason": "<简要说明>"}}
"""

        import openai
        resp = await self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )

        import json
        text = resp.choices[0].message.content or "{}"
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return {"llm_judge": 5.0}  # fallback default

        # 归一化到 [0, 1]
        scores = {k: v / 10.0 for k, v in data.items() if isinstance(v, (int, float)) and k != "reason"}
        return {"llm_judge": round(sum(scores.values()) / len(scores), 4) if scores else 5.0}


class HallucinationMetric(BaseMetric):
    """
    幻觉检测。

    检查回答中是否包含与参考资料矛盾或凭空捏造的信息。
    """

    def __init__(self, llm_client, model: str = "gpt-4o"):
        self._client = llm_client
        self._model = model

    @property
    def name(self) -> str:
        return "hallucination"

    async def score(
        self, test_case: TestCase, actual_output: str
    ) -> dict[str, float]:
        """返回 hallucination_score [0, 1]，越低越好"""
        prompt = f"""你是一个事实核查专家。请判断以下回答是否包含幻觉（即与已知事实不符或凭空捏造的信息）。

## 用户问题
{test_case.input}

## Agent 回答
{actual_output}

请以 JSON 格式回复：
{{"has_hallucination": true/false, "hallucinated_facts": ["..."], "score": <0-1，1表示无幻觉>}}
"""

        import openai
        resp = await self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )

        import json
        text = resp.choices[0].message.content or "{}"
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()

        try:
            data = json.loads(text)
            score = float(data.get("score", 0.5))
            return {"hallucination": round(score, 4)}
        except (json.JSONDecodeError, ValueError):
            return {"hallucination": 0.5}


class LatencyMetric(BaseMetric):
    """延迟指标（不依赖输出内容，只统计时间）"""

    @property
    def name(self) -> str:
        return "latency"

    async def score(
        self, test_case: TestCase, actual_output: str
    ) -> dict[str, float]:
        # 延迟由评估器在调用时记录，这里返回占位符
        return {"latency": 0.0}


# ==================== 指标工厂 ====================

def create_metric(name: str, **kwargs) -> BaseMetric:
    """创建评估指标实例"""
    metrics_map = {
        "bleu": BleuMetric,
        "rouge": RougeMetric,
        "bertscore": BertScoreMetric,
        "llm_judge": LLMJudgeMetric,
        "hallucination": HallucinationMetric,
        "latency": LatencyMetric,
    }

    cls = metrics_map.get(name)
    if cls is None:
        raise ValueError(f"Unknown metric '{name}'. Available: {list(metrics_map.keys())}")

    return cls(**kwargs)
```

---

## 5. 评估引擎 `evaluation/engine.py`

```python
"""
评估引擎：执行测试集，计算指标，生成报告。
"""

import asyncio
import logging
import statistics
import time
from evaluation.protocol import TestCase, TestResult, EvaluationReport
from evaluation.test_suite import TestSuite
from evaluation.metrics import BaseMetric

logger = logging.getLogger(__name__)


class EvaluationEngine:
    """
    评估引擎。

    Usage:
        engine = EvaluationEngine(agent_core, metrics=[llm_judge, bleu])
        report = await engine.evaluate(test_suite)
    """

    def __init__(
        self,
        agent_core,                           # AgentCore 实例（待评估对象）
        metrics: list[BaseMetric],            # 评估指标列表
        session_prefix: str = "eval",         # 会话 ID 前缀
    ):
        self._agent = agent_core
        self._metrics = metrics
        self._session_prefix = session_prefix

    async def evaluate(self, test_suite: TestSuite) -> EvaluationReport:
        """执行完整评估"""
        logger.info(f"Starting evaluation for suite '{test_suite.name}' ({len(test_suite.cases)} cases)")

        results: list[TestResult] = []
        latencies: list[float] = []

        # 并发执行测试用例（控制并发度）
        semaphore = asyncio.Semaphore(10)

        async def run_case(tc: TestCase) -> TestResult:
            async with semaphore:
                return await self._run_single_case(tc)

        tasks = [run_case(tc) for tc in test_suite.cases]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 处理异常结果
        valid_results = []
        for r in results:
            if isinstance(r, Exception):
                logger.error(f"Test case failed with exception: {r}")
                valid_results.append(TestResult(
                    test_case_id="unknown",
                    actual_output=f"ERROR: {str(r)}",
                    metrics={},
                    passed=False,
                ))
            else:
                valid_results.append(r)

        # 计算统计指标
        latencies = [r.latency_ms for r in valid_results]
        all_metrics: dict[str, list[float]] = {}
        for r in valid_results:
            for metric_name, score in r.metrics.items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(score)

        average_metrics = {k: round(statistics.mean(v), 4) for k, v in all_metrics.items()}

        passed_count = sum(1 for r in valid_results if r.passed)

        return EvaluationReport(
            test_suite_name=test_suite.name,
            total_cases=len(test_suite.cases),
            passed_cases=passed_count,
            pass_rate=round(passed_count / len(valid_results), 4) if valid_results else 0.0,
            average_metrics=average_metrics,
            latency_p50=self._percentile(latencies, 50),
            latency_p95=self._percentile(latencies, 95),
            latency_p99=self._percentile(latencies, 99),
            results=valid_results,
        )

    async def _run_single_case(self, test_case: TestCase) -> TestResult:
        """执行单条测试用例"""
        start_time = time.monotonic()

        # 调用 Agent
        session_id = f"{self._session_prefix}_{test_case.id}"
        response_text = ""

        try:
            async for chunk in self._agent.process_stream(session_id, test_case.input):
                if chunk.delta_content:
                    response_text += chunk.delta_content
        except Exception as e:
            logger.error(f"Agent error on case {test_case.id}: {e}")
            response_text = f"ERROR: {str(e)}"

        latency_ms = (time.monotonic() - start_time) * 1000

        # 计算各指标得分
        all_scores: dict[str, float] = {}
        for metric in self._metrics:
            try:
                scores = await metric.score(test_case, response_text)
                all_scores.update(scores)
            except Exception as e:
                logger.warning(f"Metric {metric.name} failed on case {test_case.id}: {e}")

        # 注入延迟指标
        all_scores["latency_ms"] = round(latency_ms, 2)

        # 综合判定是否通过
        passed = self._judge_pass(test_case, all_scores)

        return TestResult(
            test_case_id=test_case.id,
            actual_output=response_text,
            metrics=all_scores,
            passed=passed,
            latency_ms=latency_ms,
        )

    def _judge_pass(self, test_case: TestCase, scores: dict[str, float]) -> bool:
        """综合判定是否通过"""
        # 如果有期望输出，检查精确匹配或阈值匹配
        if test_case.expected_output and test_case.expected_output.strip():
            # 检查是否有指标得分低于阈值
            for metric_name, threshold in (test_case.criteria or {}).items():
                score = scores.get(metric_name)
                if score is not None and score < threshold * 0.7:  # 70% of max as pass threshold
                    return False

        # LLM Judge 分数低于 5/10 视为不通过
        llm_score = scores.get("llm_judge")
        if llm_score is not None and llm_score < 0.5:
            return False

        return True

    @staticmethod
    def _percentile(data: list[float], p: float) -> float:
        """计算百分位数"""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * p / 100
        f = int(k)
        c = f + 1
        if c >= len(sorted_data):
            return sorted_data[-1]
        d0 = sorted_data[f] * (c - k)
        d1 = sorted_data[c] * (k - f)
        return round(d0 + d1, 2)


# ==================== 多模型对比 ====================

class ModelComparisonEngine:
    """
    多模型对比引擎。

    在同一测试集上评估多个 Agent/模型，生成横向对比报告。
    """

    def __init__(self, engines: dict[str, EvaluationEngine]):
        """
        Args:
            engines: {model_name: EvaluationEngine} 映射
        """
        self._engines = engines

    async def compare(self, test_suite: TestSuite) -> "ComparisonReport":
        """执行多模型对比"""
        from evaluation.protocol import ModelComparisonReport

        reports = {}
        for model_name, engine in self._engines.items():
            logger.info(f"Evaluating model: {model_name}")
            report = await engine.evaluate(test_suite)
            reports[model_name] = report

        # 构建对比数据
        metrics_comparison: dict[str, dict[str, float]] = {}
        for model_name, report in reports.items():
            metrics_comparison[model_name] = {
                **report.average_metrics,
                "pass_rate": report.pass_rate,
                "latency_p50": report.latency_p50,
                "latency_p95": report.latency_p95,
            }

        # 找出各指标最优模型
        winner = {}
        for metric_name in next(iter(metrics_comparison.values())).keys():
            best_model = max(
                metrics_comparison,
                key=lambda m: metrics_comparison[m].get(metric_name, 0)
            )
            winner[metric_name] = best_model

        return ModelComparisonReport(
            models=list(reports.keys()),
            metrics=metrics_comparison,
            winner=winner,
        )
```

---

## 6. CI/CD 集成 `evaluation/ci.py`

```python
"""
CI/CD 集成：命令行评估工具。

Usage:
    python -m evaluation.ci --suite tests/general.json --model gpt-4o --output report.json
"""

import argparse
import json
import sys
from pathlib import Path


async def run_evaluation(args):
    """CLI 入口"""
    from evaluation.test_suite import TestSuite
    from evaluation.metrics import create_metric
    from evaluation.engine import EvaluationEngine

    # 加载测试集
    test_suite = TestSuite.from_json(args.suite)

    # 创建指标
    metrics = []
    for metric_name in args.metrics:
        kwargs = {}
        if metric_name in ("llm_judge", "hallucination"):
            import openai
            kwargs["llm_client"] = openai.AsyncOpenAI(api_key=args.api_key)
            kwargs["model"] = args.judge_model or "gpt-4o"
        metrics.append(create_metric(metric_name, **kwargs))

    # 创建 Agent（根据配置）
    from agent.core import AgentCore
    from llm.base import create_llm

    llm = create_llm(args.model, {"api_key": args.api_key})
    agent = AgentCore(llm=llm)

    # 执行评估
    engine = EvaluationEngine(agent, metrics)
    report = await engine.evaluate(test_suite)

    # 输出报告
    if args.output:
        report_data = {
            "test_suite": report.test_suite_name,
            "total_cases": report.total_cases,
            "passed_cases": report.passed_cases,
            "pass_rate": report.pass_rate,
            "average_metrics": report.average_metrics,
            "latency_p50": report.latency_p50,
            "latency_p95": report.latency_p95,
            "latency_p99": report.latency_p99,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)

    # 控制台摘要
    print(f"\n{'='*60}")
    print(f"  Evaluation Report: {report.test_suite_name}")
    print(f"{'='*60}")
    print(f"  Total Cases:   {report.total_cases}")
    print(f"  Passed:        {report.passed_cases} ({report.pass_rate * 100:.1f}%)")
    print(f"  Latency P50:   {report.latency_p50:.0f}ms")
    print(f"  Latency P95:   {report.latency_p95:.0f}ms")
    print(f"  Average Metrics:")
    for metric_name, score in report.average_metrics.items():
        print(f"    {metric_name}: {score}")
    print(f"{'='*60}\n")

    # CI 退出码：通过率低于阈值则失败
    if report.pass_rate < (args.threshold or 0.8):
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Agent Evaluation CLI")
    parser.add_argument("--suite", required=True, help="Path to test suite JSON file")
    parser.add_argument("--model", default="gpt-4o", help="Model to evaluate")
    parser.add_argument("--api-key", default="", help="API Key")
    parser.add_argument(
        "--metrics", nargs="+", default=["llm_judge"],
        help="Metrics to compute (bleu, rouge, bertscore, llm_judge, hallucination)"
    )
    parser.add_argument("--judge-model", default=None, help="Model for LLM-as-Judge")
    parser.add_argument("--output", default=None, help="Output report JSON path")
    parser.add_argument("--threshold", type=float, default=0.8, help="Pass rate threshold for CI")

    args = parser.parse_args()
    asyncio.run(run_evaluation(args))


if __name__ == "__main__":
    main()
```

---

## 7. 架构总览

```
                    ┌─────────────────────┐
                    │   Evaluation Engine │
                    │                     │
                    │  TestSuite → Cases  │──→ Agent Core (SUT)
                    │       ↓             │     ↓
                    │    Metrics          │──→ Response + Latency
                    │       ↓             │
                    │  EvaluationReport   │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                 ▼
     ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
     │ BleuMetric  │  │ LLMJudge    │  │ Hallucination│
     │ RougeMetric │  │ Metric      │  │ Metric       │
     │ BertScore   │  │             │  │              │
     └─────────────┘  └─────────────┘  └─────────────┘

                    ┌─────────────────────┐
                    │ Comparison Engine   │
                    │                     │
                    │ Multi-model → Report│──→ ModelComparisonReport
                    └─────────────────────┘

                    ┌─────────────────────┐
                    │     CI/CD CLI       │
                    │                     │
                    │ pytest / GitHub Actions │──→ Pass/Fail + JSON Report
                    └─────────────────────┘
```

---

## 8. 设计总结

| 特性 | 实现方式 |
|------|---------|
| **多维度评估** | BLEU/ROUGE/BERTScore（传统）+ LLM Judge/Hallucination（语义） |
| **LLM-as-Judge** | GPT-4o 作为裁判，按自定义标准打分 |
| **幻觉检测** | 专门的 LLM prompt 检查事实一致性 |
| **性能监控** | P50/P95/P99 延迟统计 + Token 消耗追踪 |
| **多模型对比** | ComparisonEngine 横向对比多个 Agent/模型 |
| **CI/CD 集成** | CLI 工具 + JSON 报告 + 通过率阈值判定 |
| **测试集管理** | JSON 格式，支持标签过滤、动态构建 |

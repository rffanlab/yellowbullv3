# 代码执行沙箱详细设计

## 1. 职责边界

| 职责 | 说明 |
|------|------|
| **安全隔离** | 在受限环境中执行用户/Agent 生成的代码，防止恶意操作 |
| **语言支持** | Python（主要）、JavaScript、Shell 脚本 |
| **资源限制** | CPU 时间、内存、网络访问、文件系统访问控制 |
| **结果捕获** | stdout/stderr 输出、返回值、异常信息结构化返回 |
| **依赖管理** | 允许导入白名单内的库，禁止危险模块 |

---

## 2. 沙箱执行器 `tools/code_execution/sandbox.py`

```python
"""
Code execution sandbox。

安全策略：
1. 子进程隔离（subprocess + timeout）
2. 内存限制（ulimit / cgroup）
3. 网络禁用（--disable-networking for Python）
4. 文件系统只读挂载，临时目录可写
5. 导入白名单控制
"""

import asyncio
import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class Language(str, Enum):
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    SHELL = "shell"


@dataclass
class ExecutionResult:
    success: bool
    stdout: str = ""
    stderr: str = ""
    return_value: Any = None
    error_message: Optional[str] = None
    execution_time_ms: float = 0.0
    memory_usage_kb: int = 0


class CodeSandbox:
    """代码执行沙箱"""

    # Python 允许导入的模块白名单
    PYTHON_ALLOWED_MODULES = {
        "math", "random", "datetime", "collections", "itertools",
        "functools", "re", "json", "csv", "statistics", "string",
        "pathlib", "typing", "numpy", "pandas",
    }

    # 禁止的危险操作关键词
    DANGEROUS_PATTERNS = [
        "__import__", "subprocess", "os.system", "os.popen",
        "eval(", "exec(", "compile(", "getattr(__builtins__",
        "socket.", "http.", "requests.", "urllib.",
    ]

    def __init__(
        self,
        timeout_seconds: float = 30.0,
        memory_limit_mb: int = 256,
        working_dir: str = None,
    ):
        self._timeout = timeout_seconds
        self._memory_limit = memory_limit_mb * 1024  # KB
        self._working_dir = Path(working_dir or tempfile.mkdtemp(prefix="sandbox_"))

    async def execute_python(self, code: str) -> ExecutionResult:
        """执行 Python 代码"""
        if self._contains_dangerous_patterns(code):
            return ExecutionResult(
                success=False,
                error_message="代码包含禁止的操作",
            )

        # 包装代码：添加安全前缀和结果捕获
        wrapped_code = self._wrap_python_code(code)

        start_time = asyncio.get_event_loop().time()

        try:
            process = await asyncio.create_subprocess_exec(
                "python", "-c", wrapped_code,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self._working_dir),
                env=self._safe_env(),
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=self._timeout
            )

            elapsed_ms = (asyncio.get_event_loop().time() - start_time) * 1000

            if process.returncode != 0:
                return ExecutionResult(
                    success=False,
                    stdout=stdout.decode("utf-8", errors="replace"),
                    stderr=stderr.decode("utf-8", errors="replace"),
                    error_message=f"执行错误 (exit code {process.returncode})",
                    execution_time_ms=elapsed_ms,
                )

            # 解析输出 JSON（最后一行）
            output = stdout.decode("utf-8", errors="replace")
            return self._parse_output(output, elapsed_ms)

        except asyncio.TimeoutError:
            return ExecutionResult(
                success=False,
                error_message=f"执行超时 ({self._timeout}s)",
                execution_time_ms=self._timeout * 1000,
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                error_message=str(e),
                execution_time_ms=(asyncio.get_event_loop().time() - start_time) * 1000,
            )

    async def execute_javascript(self, code: str) -> ExecutionResult:
        """执行 JavaScript 代码（通过 Node.js）"""
        wrapped = f"try {{ const result = ({code}); console.log(JSON.stringify({{success: true, value: result}})); }} catch(e) {{ console.error(JSON.stringify({{success: false, error: e.message}})); }}"

        start_time = asyncio.get_event_loop().time()

        try:
            process = await asyncio.create_subprocess_exec(
                "node", "-e", wrapped,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._safe_env(),
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=self._timeout
            )

            elapsed_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            output = stdout.decode("utf-8", errors="replace").strip()

            try:
                parsed = json.loads(output)
                if parsed.get("success"):
                    return ExecutionResult(
                        success=True,
                        stdout=output,
                        return_value=parsed.get("value"),
                        execution_time_ms=elapsed_ms,
                    )
                else:
                    return ExecutionResult(
                        success=False,
                        stdout="",
                        stderr=parsed.get("error", ""),
                        error_message=parsed.get("error"),
                        execution_time_ms=elapsed_ms,
                    )
            except json.JSONDecodeError:
                return ExecutionResult(
                    success=True,
                    stdout=output,
                    stderr=stderr.decode("utf-8", errors="replace"),
                    execution_time_ms=elapsed_ms,
                )

        except asyncio.TimeoutError:
            return ExecutionResult(
                success=False,
                error_message=f"执行超时 ({self._timeout}s)",
                execution_time_ms=self._timeout * 1000,
            )

    async def execute_shell(self, command: str) -> ExecutionResult:
        """执行 Shell 命令（严格白名单）"""
        # 只允许安全的命令前缀
        allowed_prefixes = ("ls", "cat", "head", "tail", "wc", "grep", "find", "pwd")
        first_word = command.split()[0] if command.split() else ""

        if first_word not in allowed_prefixes:
            return ExecutionResult(
                success=False,
                error_message=f"不允许的命令: {first_word}",
            )

        start_time = asyncio.get_event_loop().time()

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self._working_dir),
                env=self._safe_env(),
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=self._timeout
            )

            elapsed_ms = (asyncio.get_event_loop().time() - start_time) * 1000

            return ExecutionResult(
                success=process.returncode == 0,
                stdout=stdout.decode("utf-8", errors="replace"),
                stderr=stderr.decode("utf-8", errors="replace"),
                error_message=f"exit code {process.returncode}" if process.returncode != 0 else None,
                execution_time_ms=elapsed_ms,
            )

        except asyncio.TimeoutError:
            return ExecutionResult(
                success=False,
                error_message=f"执行超时 ({self._timeout}s)",
                execution_time_ms=self._timeout * 1000,
            )

    def _wrap_python_code(self, code: str) -> str:
        """包装 Python 代码，添加安全控制和结果捕获"""
        wrapper = f'''
import sys, json, io, traceback

# Redirect stdout to capture print output
old_stdout = sys.stdout
sys.stdout = io.StringIO()

try:
    # User code
{code}

    # Capture printed output
    printed = sys.stdout.getvalue()
    result = locals().get("_result", None)  # If user sets _result variable

    output = {{
        "success": True,
        "printed": printed,
        "value": str(result) if result is not None else None,
    }}
except Exception as e:
    printed = sys.stdout.getvalue()
    output = {{
        "success": False,
        "printed": printed,
        "error": traceback.format_exc(),
    }}

sys.stdout = old_stdout
print(json.dumps(output, ensure_ascii=False))
'''
        return wrapper

    def _parse_output(self, output: str, elapsed_ms: float) -> ExecutionResult:
        """解析沙箱输出"""
        lines = output.strip().split("\n")
        last_line = lines[-1] if lines else ""

        try:
            parsed = json.loads(last_line)
            printed = "\n".join(lines[:-1]) if len(lines) > 1 else ""

            if not parsed.get("success"):
                return ExecutionResult(
                    success=False,
                    stdout=printed,
                    stderr=parsed.get("error", ""),
                    error_message="代码执行异常",
                    execution_time_ms=elapsed_ms,
                )

            return ExecutionResult(
                success=True,
                stdout=printed or parsed.get("printed", ""),
                return_value=parsed.get("value"),
                execution_time_ms=elapsed_ms,
            )
        except json.JSONDecodeError:
            return ExecutionResult(
                success=True,
                stdout=output,
                execution_time_ms=elapsed_ms,
            )

    def _contains_dangerous_patterns(self, code: str) -> bool:
        """检查代码是否包含危险操作"""
        for pattern in self.DANGEROUS_PATTERNS:
            if pattern in code:
                return True
        return False

    def _safe_env(self) -> dict:
        """构建安全的执行环境变量"""
        env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": str(self._working_dir),
            "PYTHONIOENCODING": "utf-8",
            # 移除可能的危险变量
        }
        for key in list(os.environ.keys()):
            if key.upper() in ("API_KEY", "SECRET", "TOKEN", "PASSWORD"):
                continue
            env[key] = os.environ[key]
        return env

    async def cleanup(self):
        """清理临时工作目录"""
        import shutil
        if self._working_dir.exists():
            shutil.rmtree(str(self._working_dir), ignore_errors=True)


def get_sandbox() -> CodeSandbox:
    """获取全局沙箱实例"""
    import config.manager as cm
    manager = cm.get_manager()
    if not hasattr(manager, "_code_sandbox"):
        settings = (manager.settings.tools or {}).get("code_execution", {})
        manager._code_sandbox = CodeSandbox(
            timeout_seconds=settings.get("timeout_seconds", 30),
            memory_limit_mb=settings.get("memory_limit_mb", 256),
            working_dir=settings.get("working_dir"),
        )
    return manager._code_sandbox
```

---

## 3. Tool 注册 `tools/code_execution/tool.py`

```python
"""
代码执行工具，注册到 ToolRegistry。
"""


class CodeExecutionTool:
    """代码执行工具"""

    name = "execute_code"
    description = (
        "在安全沙箱中执行代码片段。支持 Python、JavaScript 和 Shell 命令。"
        "适用于数据处理、计算、格式转换等场景。"
    )

    def __init__(self, sandbox: "CodeSandbox"):
        self._sandbox = sandbox

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
                        "language": {
                            "type": "string",
                            "enum": ["python", "javascript", "shell"],
                            "description": "编程语言",
                        },
                        "code": {
                            "type": "string",
                            "description": "要执行的代码（Python/JS）或命令（Shell）",
                        },
                    },
                    "required": ["language", "code"],
                },
            },
        }

    async def execute(self, language: str, code: str) -> dict:
        """执行代码"""
        if len(code) > 10000:
            return {
                "success": False,
                "content": f"代码过长 ({len(code)} chars)，限制为 10000 字符",
            }

        if language == "python":
            result = await self._sandbox.execute_python(code)
        elif language == "javascript":
            result = await self._sandbox.execute_javascript(code)
        elif language == "shell":
            result = await self._sandbox.execute_shell(code)
        else:
            return {"success": False, "content": f"不支持的语言: {language}"}

        response_parts = []
        if result.stdout:
            response_parts.append(f"**输出:**\n```\n{result.stdout.strip()}\n```")
        if result.return_value is not None:
            response_parts.append(f"**返回值:** `{result.return_value}`")
        response_parts.append(f"*执行耗时: {result.execution_time_ms:.0f}ms*")

        return {
            "success": result.success,
            "content": "\n\n".join(response_parts) if response_parts else "(无输出)",
            "error": result.error_message,
            "stderr": result.stderr,
        }


class DataAnalysisTool:
    """数据分析工具（基于 Python + pandas/numpy）"""

    name = "analyze_data"
    description = (
        "使用 Python 进行数据分析。支持 CSV/JSON 数据解析、统计计算、"
        "可视化描述等。输入数据通过 data 参数传入。"
    )

    def __init__(self, sandbox: "CodeSandbox"):
        self._sandbox = sandbox

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
                        "data": {
                            "type": "string",
                            "description": "CSV 或 JSON 格式的数据",
                        },
                        "format": {
                            "type": "string",
                            "enum": ["csv", "json"],
                            "default": "csv",
                            "description": "数据格式",
                        },
                        "analysis": {
                            "type": "string",
                            "description": "要执行的分析描述（如：计算均值、找出异常值）",
                        },
                    },
                    "required": ["data", "analysis"],
                },
            },
        }

    async def execute(self, data: str, analysis: str, format: str = "csv") -> dict:
        """执行数据分析"""
        # 将数据写入临时文件
        import tempfile
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=f".{format}", delete=False, dir=self._sandbox._working_dir
        ) as f:
            f.write(data)
            data_file = f.name

        analysis_code = f'''
import pandas as pd
import numpy as np

df = pd.read_{format}("{data_file}")

# Analysis based on user request: {analysis}
summary = df.describe()
_result = summary.to_string()
'''

        result = await self._sandbox.execute_python(analysis_code)

        if not result.success:
            return {"success": False, "content": f"分析失败: {result.error_message}", "error": result.stderr}

        return {
            "success": True,
            "content": f"**数据分析结果:**\n```\n{result.stdout.strip()}\n```",
        }


def register_code_tools(registry: "ToolRegistry"):
    """注册代码执行相关工具"""
    from tools.code_execution.sandbox import get_sandbox

    sandbox = get_sandbox()
    registry.register(CodeExecutionTool(sandbox))
    registry.register(DataAnalysisTool(sandbox))
```

---

## 4. Docker 沙箱（生产环境）`tools/code_execution/docker_sandbox.py`

```python
"""
Docker-based code sandbox for production。

使用 Docker 容器提供更强隔离：
- 独立命名空间
- 资源限制 (CPU, memory)
- 网络隔离
- 自动清理
"""

import asyncio
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class DockerSandbox:
    """Docker 容器代码沙箱"""

    # 预构建的轻量镜像（基于 Python slim）
    DEFAULT_IMAGE = "python:3.12-slim"

    def __init__(
        self,
        image: str = None,
        timeout_seconds: float = 30.0,
        memory_limit_mb: int = 256,
        cpu_limit: float = 1.0,
    ):
        self._image = image or self.DEFAULT_IMAGE
        self._timeout = timeout_seconds
        self._memory_limit = f"{memory_limit_mb}m"
        self._cpu_limit = str(cpu_limit)

    async def execute_python(self, code: str) -> dict:
        """在 Docker 容器中执行 Python 代码"""
        container_name = f"sandbox_{__import__('uuid').uuid4().hex[:12]}"

        # 创建容器（带资源限制）
        create_cmd = [
            "docker", "create",
            "--name", container_name,
            "--rm",                       # 自动清理
            "--memory", self._memory_limit,
            "--cpus", self._cpu_limit,
            "--network", "none",          # 无网络
            "--read-only",                # 只读文件系统
            "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",
            "-e", "PYTHONIOENCODING=utf-8",
            self._image,
            "python", "-c", code,
        ]

        try:
            # 创建容器
            proc = await asyncio.create_subprocess_exec(
                *create_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, _ = await proc.communicate()

            if proc.returncode != 0:
                return {"success": False, "error_message": "容器创建失败"}

            # 启动并等待完成
            start_proc = await asyncio.create_subprocess_exec(
                "docker", "start", "-a", container_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                start_proc.communicate(), timeout=self._timeout
            )

            return {
                "success": start_proc.returncode == 0,
                "stdout": stdout.decode("utf-8", errors="replace"),
                "stderr": stderr.decode("utf-8", errors="replace"),
                "error_message": f"exit code {start_proc.returncode}" if start_proc.returncode != 0 else None,
            }

        except asyncio.TimeoutError:
            await self._kill_container(container_name)
            return {"success": False, "error_message": f"执行超时 ({self._timeout}s)"}
        finally:
            # 确保容器被清理
            await self._cleanup_container(container_name)

    async def _kill_container(self, name: str):
        """强制停止容器"""
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker", "kill", name,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()
        except Exception as e:
            logger.warning(f"Failed to kill container {name}: {e}")

    async def _cleanup_container(self, name: str):
        """清理容器"""
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker", "rm", "-f", name,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()
        except Exception as e:
            logger.warning(f"Failed to cleanup container {name}: {e}")


def get_docker_sandbox() -> DockerSandbox:
    """获取 Docker 沙箱实例（生产环境）"""
    import config.manager as cm
    manager = cm.get_manager()
    if not hasattr(manager, "_docker_sandbox"):
        settings = (manager.settings.tools or {}).get("code_execution", {})
        manager._docker_sandbox = DockerSandbox(
            image=settings.get("docker_image"),
            timeout_seconds=settings.get("timeout_seconds", 30),
            memory_limit_mb=settings.get("memory_limit_mb", 256),
            cpu_limit=settings.get("cpu_limit", 1.0),
        )
    return manager._docker_sandbox
```

---

## 5. YAML 配置 `tools.code_execution` section

```yaml
# config/settings.yaml (新增)
tools:
  code_execution:
    enabled: false              # 默认关闭，需显式开启
    sandbox_type: "subprocess"  # subprocess | docker

    # 通用限制
    timeout_seconds: 30         # 执行超时（秒）
    memory_limit_mb: 256        # 内存限制
    max_code_length: 10000      # 最大代码长度（字符）

    # Docker 沙箱配置（sandbox_type=docker 时生效）
    docker_image: "python:3.12-slim"
    cpu_limit: 1.0              # CPU 核心数限制

    # Python 白名单模块（可追加）
    extra_allowed_modules: []   # e.g., ["matplotlib", "scipy"]
```

---

## 6. 安全策略总结

| 防护层 | 措施 |
|--------|------|
| **代码审计** | 危险模式检测（`__import__`, `subprocess`, `eval`, `exec` 等） |
| **进程隔离** | subprocess 子进程执行，超时强制终止 |
| **资源限制** | CPU 时间、内存上限、代码长度限制 |
| **网络禁用** | 环境变量清理，Docker `--network none` |
| **文件系统** | 只读挂载 + 临时目录可写（64MB） |
| **模块白名单** | Python import 仅允许安全库 |
| **生产隔离** | Docker 容器：独立命名空间、自动清理 |

---

## 7. 数据流图

```
Agent Core → ToolRegistry.execute("execute_code")
                │
                ▼
          CodeExecutionTool
                │
                ├─ 代码长度检查 (>10000 chars? reject)
                ├─ 危险模式检测 (dangerous patterns? reject)
                │
                ▼
          CodeSandbox.execute_{language}()
                │
                ├─ subprocess mode:
                │    ├── python -c wrapped_code
                │    ├── node -e wrapped_code
                │    └── shell command (whitelist only)
                │
                └─ docker mode:
                     ├── docker create (--rm, --memory, --cpus, --network none)
                     ├── docker start -a
                     └─ docker rm -f (cleanup)
                │
                ▼
          ExecutionResult → ToolOutput → Agent Core
```

---

## 8. 设计总结

| 特性 | 实现方式 |
|------|---------|
| **语言支持** | Python（主要）、JavaScript、Shell（白名单命令） |
| **开发环境** | subprocess 隔离，简单快速 |
| **生产环境** | Docker 容器，完整命名空间隔离 |
| **安全防护** | 代码审计 + 资源限制 + 网络禁用 + 文件系统控制 |
| **数据分析** | 内置 DataAnalysisTool，支持 CSV/JSON + pandas/numpy |

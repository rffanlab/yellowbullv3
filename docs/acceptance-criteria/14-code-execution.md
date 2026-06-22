# 代码执行沙箱 — 验收标准

## SANDBOX-01: Python 代码执行

**优先级**: P0

- **Given** `SandboxExecutor` 已初始化（Docker container）
- **When** 调用 `execute(code="print(1+1)", language="python")`
- **Then** 返回 `ExecutionResult(stdout="2\n", stderr="", exit_code=0, duration_ms=N)`；容器不被污染
- **验证方式**: 集成测试（Docker）

## SANDBOX-02: Shell 命令执行

**优先级**: P1

- **Given** `SandboxExecutor` 已初始化
- **When** 调用 `execute(code="echo hello", language="shell")`
- **Then** 返回正确 stdout；容器内 shell 环境可用（bash/sh）
- **验证方式**: 集成测试

## SANDBOX-03: 超时控制

**优先级**: P0

- **Given** 配置 `timeout=5` 秒
- **When** 执行无限循环代码 `while True: pass`
- **Then** 5s 后进程被终止；返回 `ExecutionResult(exit_code=-1, error="Timeout")`；容器保持可用
- **验证方式**: 集成测试

## SANDBOX-04: 内存限制

**优先级**: P1

- **Given** Docker container 配置 memory limit（如 256MB）
- **When** 执行 `x = "a" * (512 * 1024 * 1024)`（分配 512MB）
- **Then** 进程被 OOM kill；返回错误结果；容器不崩溃
- **验证方式**: 集成测试

## SANDBOX-05: CPU 限制

**优先级**: P1

- **Given** Docker container 配置 cpu quota（如 0.5 cores）
- **When** 执行 CPU 密集型代码
- **Then** CPU 使用率不超过限制；不影响宿主机性能
- **验证方式**: 手动验证 — docker stats

## SANDBOX-06: 网络隔离

**优先级**: P1

- **Given** Docker container 配置 `--network none`（或自定义 network）
- **When** 执行 `import socket; s.connect(("8.8.8.8", 80))`
- **Then** 连接失败；容器无法访问外部网络（除非显式允许）
- **验证方式**: 集成测试

## SANDBOX-07: 文件系统隔离

**优先级**: P1

- **Given** Docker container 使用临时 filesystem
- **When** 执行 `open("/etc/passwd").read()` / `os.listdir("/")`
- **Then** 只能访问容器内最小化文件系统；宿主机文件不可达；写入操作在容器销毁后丢失
- **验证方式**: 集成测试

## SANDBOX-08: 危险操作拦截

**优先级**: P1

- **Given** `CodeExecutionGuard` 已启用
- **When** 代码包含 `rm -rf /`、`import os; os.unmount(...)` 等危险模式
- **Then** 在执行前被检测到并拒绝；返回错误结果，不启动容器
- **验证方式**: 单元测试 — 已知危险 payload

## SANDBOX-09: 依赖安装

**优先级**: P2

- **Given** 代码需要第三方库（如 `import pandas`）
- **When** 调用 `execute(code, language="python", dependencies=["pandas"])`
- **Then** pip install 指定依赖后执行；安装过程计入 timeout；失败时返回错误
- **验证方式**: 集成测试

## SANDBOX-10: 输出截断

**优先级**: P1

- **Given** 配置 `max_output_bytes=1024*1024`（1MB）
- **When** 代码输出超过限制
- **Then** stdout/stderr 被截断；返回结果包含 `truncated=True` 标记
- **验证方式**: 集成测试

## SANDBOX-11: Container 复用与清理

**优先级**: P2

- **Given** 多次执行请求
- **When** 容器池管理容器生命周期
- **Then** 空闲容器被复用（减少启动开销）；超过 idle_timeout 的容器自动销毁；并发数不超过 max_containers
- **验证方式**: 集成测试 — 断言容器数量

## SANDBOX-12: ExecutionResult 格式

**优先级**: P0

- **Given** 任意执行结果
- **When** 返回 `ExecutionResult`
- **Then** 包含 `stdout`、`stderr`、`exit_code`、`duration_ms`；成功时 exit_code=0，失败时非 0
- **验证方式**: 单元测试

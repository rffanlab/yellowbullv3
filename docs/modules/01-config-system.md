# 配置系统详细设计

## 1. 设计目标

| 目标 | 说明 |
|------|------|
| **集中管理** | 所有运行时参数统一从配置文件加载，零硬编码 |
| **热更新** | 修改 YAML 后无需重启服务，配置自动生效 |
| **环境变量注入** | 支持 `${ENV_VAR}` 引用，敏感信息不进代码库 |
| **类型安全** | 加载结果映射为强类型 dataclass，IDE 可补全 |
| **分层覆盖** | default.yaml → env.yaml → CLI args，后者优先级高 |
| **变更通知** | 配置变更后通过回调/事件通知依赖方 |

---

## 2. 文件结构

```
config/
├── __init__.py              # 导出 Settings, load_settings, ConfigManager
├── settings.py              # Dataclass 定义 + 解析逻辑
├── manager.py               # ConfigManager：热更新 + 变更通知
├── default.yaml             # 默认配置（提交到仓库）
├── .env.example             # 环境变量模板（不提交真实值）
└── environments/
    ├── development.yaml     # 开发环境覆盖
    ├── staging.yaml         # 预发布覆盖
    └── production.yaml      # 生产覆盖
```

使用方式：
```bash
# 开发环境
python main.py --env development        # 加载 default.yaml + environments/development.yaml

# 生产环境
python main.py --env production         # 加载 default.yaml + environments/production.yaml
```

---

## 3. Dataclass 定义 `settings.py`

### 3.1 完整类型树

```python
from dataclasses import dataclass, field
from typing import Any
from enum import Enum


# ==================== LLM 配置 ====================

@dataclass
class ProviderConfig:
    """单个 provider 的参数"""
    api_key: str = ""
    model: str = ""
    temperature: float = 0.7
    max_tokens: int = 4096
    base_url: str | None = None       # OpenAI/Azure/Ollama 兼容
    deployment_name: str | None = None # Azure 专用
    endpoint: str | None = None        # Azure 专用
    extra: dict[str, Any] = field(default_factory=dict)

@dataclass
class LLMConfig:
    provider: str = "openai"           # openai | anthropic | ollama | azure
    providers: dict[str, ProviderConfig] = field(default_factory=dict)
    fallback_providers: list[str] = field(default_factory=list)  # 故障降级链

    @property
    def active(self) -> ProviderConfig:
        return self.providers.get(self.provider, ProviderConfig())


# ==================== Agent 配置 ====================

@dataclass
class AgentConfig:
    system_prompt: str = ""
    context_window: int = 48               # 滑动窗口消息数
    max_tool_calls_per_turn: int = 4       # 单轮并行工具上限
    max_chain_depth: int = 5               # LLM → tool → LLM 最大循环次数
    tool_retry_limit: int = 3              # 单个工具失败重试次数
    total_timeout_seconds: float = 60.0    # 单次请求总超时
    enable_streaming: bool = True          # 是否启用流式输出


# ==================== 工具配置 ====================

@dataclass
class BuiltinToolConfig:
    enabled: bool = True
    settings: dict[str, Any] = field(default_factory=dict)

@dataclass
class ToolsConfig:
    builtin: dict[str, BuiltinToolConfig] = field(default_factory=dict)


# ==================== 会话配置 ====================

@dataclass
class SessionConfig:
    storage_backend: str = "memory"        # memory | redis | sqlite
    max_messages: int = 200                # 单会话最大消息数（硬上限）
    ttl_seconds: int = 86400              # 空闲会话过期时间（秒）
    cleanup_interval_seconds: int = 300   # 清理任务间隔


# ==================== 服务配置 ====================

@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1                       # uvicorn worker 数
    cors_origins: list[str] = field(default_factory=lambda: ["*"])
    api_prefix: str = "/api"


# ==================== 日志配置 ====================

@dataclass
class LoggingConfig:
    level: str = "INFO"                    # DEBUG | INFO | WARNING | ERROR
    format: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    file_path: str = "./logs/agent.log"
    max_bytes: int = 10 * 1024 * 1024     # 10MB 轮转
    backup_count: int = 5


# ==================== 根配置 ====================

@dataclass
class Settings:
    llm: LLMConfig = field(default_factory=LLMConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def model_dump(self) -> dict[str, Any]:
        """序列化为 dict（用于调试/监控暴露）"""
        import dataclasses
        return dataclasses.asdict(self)
```

### 3.2 设计考量

| 决策 | 理由 |
|------|------|
| `LLMConfig.providers` 存全部 provider，用 `provider` 字段切换 | 一次配置多模型，热更新时直接改 `provider` 即可切换，无需重启重建 LLM 实例 |
| `fallback_providers` 降级链 | 主 provider 故障时自动切备用，提高可用性 |
| `BuiltinToolConfig.settings` 泛型 dict | 不同工具参数差异大，用 dict 避免频繁改 dataclass |
| `SessionConfig.storage_backend` 可配 | MVP 用 memory，后续切 redis/sqlite 只需改配置 + 实现对应 backend |

---

## 4. YAML 配置文件

### 4.1 `default.yaml`

```yaml
llm:
  provider: openai
  fallback_providers: [ollama]
  providers:
    openai:
      api_key: "${OPENAI_API_KEY}"
      model: "gpt-4o"
      temperature: 0.7
      max_tokens: 4096

    anthropic:
      api_key: "${ANTHROPIC_API_KEY}"
      model: "claude-sonnet-4-20250514"
      temperature: 0.7
      max_tokens: 4096

    ollama:
      base_url: "http://localhost:11434"
      model: "qwen2.5-72b"
      temperature: 0.7
      max_tokens: 4096

    azure:
      api_key: "${AZURE_API_KEY}"
      endpoint: "${AZURE_ENDPOINT}"
      deployment_name: "gpt-4o"
      temperature: 0.7
      max_tokens: 4096

agent:
  system_prompt: |
    你是一个智能助手，可以帮助用户回答问题、执行任务。
    根据用户需求选择合适的工具，如果没有合适的工具则直接回答。
  context_window: 48
  max_tool_calls_per_turn: 4
  max_chain_depth: 5
  tool_retry_limit: 3
  total_timeout_seconds: 60
  enable_streaming: true

tools:
  builtin:
    current_time:
      enabled: true
    calculator:
      enabled: true
    web_search:
      enabled: true
      settings:
        engine: duckduckgo
        api_key: "${SEARCH_API_KEY}"
        max_results: 5

session:
  storage_backend: memory
  max_messages: 200
  ttl_seconds: 86400
  cleanup_interval_seconds: 300

server:
  host: "0.0.0.0"
  port: 8000
  workers: 1
  cors_origins: ["*"]
  api_prefix: "/api"

logging:
  level: INFO
  file_path: "./logs/agent.log"
  max_bytes: 10485760
  backup_count: 5
```

### 4.2 `environments/development.yaml`（覆盖示例）

```yaml
llm:
  provider: ollama          # 开发用本地模型省钱

logging:
  level: DEBUG              # 开发开 debug
```

---

## 5. 配置加载逻辑 `settings.py`

### 5.1 YAML → Dataclass 映射器

```python
import yaml
from pathlib import Path
from typing import Any, Type
import dataclasses


def _resolve_env_vars(value: Any) -> Any:
    """递归解析 ${ENV_VAR} 引用"""
    import os
    import re
    if isinstance(value, str):
        def replacer(match):
            env_key = match.group(1)
            return os.getenv(env_key, match.group(0))   # 未找到则保留原样
        return re.sub(r"\$\{(\w+)\}", replacer, value)
    elif isinstance(value, dict):
        return {k: _resolve_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_resolve_env_vars(item) for item in value]
    return value


def _dict_to_dataclass(cls: Type, data: dict[str, Any]) -> Any:
    """递归将 dict 转换为 dataclass 实例"""
    if not dataclasses.is_dataclass(cls):
        return data

    field_types = {f.name: f.type for f in dataclasses.fields(cls)}
    kwargs = {}

    for key, value in data.items():
        if key not in field_types:
            continue   # 忽略多余字段（向前兼容）

        target_type = field_types[key]

        # dict[str, X] → 遍历转换
        if hasattr(target_type, "__origin__") and target_type.__origin__ is dict:
            value_type = list(target_type.__args__)[1]   # dict[str, X][1]
            if dataclasses.is_dataclass(value_type):
                value = {k: _dict_to_dataclass(value_type, v) for k, v in (value or {}).items()}
            else:
                value = {k: _resolve_env_vars(v) for k, v in (value or {}).items()}

        # list[X] → 遍历转换
        elif hasattr(target_type, "__origin__") and target_type.__origin__ is list:
            item_type = target_type.__args__[0]
            if dataclasses.is_dataclass(item_type):
                value = [_dict_to_dataclass(item_type, v) for v in (value or [])]

        # 嵌套 dataclass
        elif dataclasses.is_dataclass(target_type) and isinstance(value, dict):
            value = _dict_to_dataclass(target_type, value)

        kwargs[key] = value

    return cls(**kwargs)


def _deep_merge(base: dict, override: dict) -> dict:
    """深度合并：override 覆盖 base，嵌套 dict 递归合并"""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_settings(
    default_path: str = "config/default.yaml",
    env_name: str | None = None,
) -> Settings:
    """
    加载配置：default → environment override → 合并

    Args:
        default_path: 默认配置文件路径
        env_name: 环境名（如 development），会加载 environments/{env_name}.yaml
    """
    # 1. 加载默认配置
    with open(default_path) as f:
        base = yaml.safe_load(f)

    # 2. 加载环境覆盖
    if env_name:
        env_path = Path(f"config/environments/{env_name}.yaml")
        if env_path.exists():
            with open(env_path) as f:
                override = yaml.safe_load(f)
            base = _deep_merge(base, override)

    # 3. 解析环境变量
    base = _resolve_env_vars(base)

    # 4. dict → Settings dataclass
    return _dict_to_dataclass(Settings, base)
```

### 5.2 设计考量

| 决策 | 理由 |
|------|------|
| `_deep_merge` 而非全量替换 | 环境覆盖只需写差异部分，不用复制整个 default.yaml |
| 忽略多余字段（向前兼容） | 新配置加了字段后，旧 YAML 不会报错 |
| `${ENV_VAR}` 找不到时保留原样 | 启动时打印警告但不停止，方便排查 |

---

## 6. 热更新管理器 `manager.py`

### 6.1 核心设计思路

```
┌──────────────┐    watch file change     ┌─────────────────┐
│  YAML File   │ ─────────────────────→   │  ConfigManager  │
└──────────────┘                          │                 │
                                          │  1. reload()    │
                                          │  2. diff old↔new│
                                          │  3. notify()    │
                                          └───────┬─────────┘
                                                  │ on_change
                                                  ▼
                                    ┌─────────────────────────┐
                                    │     订阅方回调           │
                                    │  • Agent.reload_config()│
                                    │  • ToolRegistry.refresh()│
                                    │  • LLM.switch_provider()│
                                    └─────────────────────────┘
```

### 6.2 实现

```python
import asyncio
import dataclasses
import hashlib
from pathlib import Path
from typing import Any, Callable
from datetime import datetime


class ConfigManager:
    """
    配置管理器：加载 + 热更新 + 变更通知

    Usage:
        manager = ConfigManager("config/default.yaml", env_name="development")
        settings = manager.settings          # 当前配置（只读属性）

        # 订阅变更
        @manager.on_change("llm.provider")
        def on_llm_change(old, new):
            print(f"LLM provider: {old} → {new}")

        # 启动文件监听（后台任务）
        await manager.start_watching()
    """

    def __init__(self, default_path: str, env_name: str | None = None):
        self._default_path = Path(default_path)
        self._env_name = env_name
        self._settings = self._load()
        self._file_hash = self._compute_hash()
        self._callbacks: list[Callable] = []           # 全局回调
        self._path_callbacks: dict[str, list[Callable]] = {}  # 路径级回调
        self._watch_task: asyncio.Task | None = None
        self._poll_interval = 2.0                      # 轮询间隔（秒）

    @property
    def settings(self):
        return self._settings

    # ==================== 加载 ====================

    def _load(self) -> "Settings":   # forward ref
        from config.settings import load_settings
        return load_settings(
            default_path=str(self._default_path),
            env_name=self._env_name,
        )

    def reload(self):
        """手动重载配置"""
        new_settings = self._load()
        self._notify_changes(new_settings)
        self._settings = new_settings
        self._file_hash = self._compute_hash()

    # ==================== 文件监听 ====================

    def _compute_hash(self) -> str:
        """计算所有配置文件的 hash"""
        import hashlib
        h = hashlib.sha256()
        for path in [self._default_path]:
            if path.exists():
                h.update(path.read_bytes())
        if self._env_name:
            env_path = Path(f"config/environments/{self._env_name}.yaml")
            if env_path.exists():
                h.update(env_path.read_bytes())
        return h.hexdigest()

    async def start_watching(self, poll_interval: float | None = None):
        """启动后台文件监听（轮询方式，跨平台兼容）"""
        interval = poll_interval or self._poll_interval
        self._watch_task = asyncio.create_task(self._watch_loop(interval))

    async def _watch_loop(self, interval: float):
        while True:
            await asyncio.sleep(interval)
            try:
                current_hash = self._compute_hash()
                if current_hash != self._file_hash:
                    print(f"[ConfigManager] Config file changed, reloading...")
                    new_settings = self._load()
                    self._notify_changes(new_settings)
                    self._settings = new_settings
                    self._file_hash = current_hash
            except Exception as e:
                print(f"[ConfigManager] Failed to reload config: {e}")

    async def stop_watching(self):
        if self._watch_task:
            self._watch_task.cancel()
            try:
                await self._watch_task
            except asyncio.CancelledError:
                pass

    # ==================== 变更检测 ====================

    @staticmethod
    def _diff_settings(old: Any, new: Any, path: str = "") -> list[tuple[str, Any, Any]]:
        """递归比较两个 settings，返回变更路径列表"""
        changes = []
        if not dataclasses.is_dataclass(old) or not dataclasses.is_dataclass(new):
            if old != new:
                changes.append((path, old, new))
            return changes

        for f in dataclasses.fields(old):
            field_path = f"{path}.{f.name}" if path else f.name
            old_val = getattr(old, f.name)
            new_val = getattr(new, f.name)
            changes.extend(ConfigManager._diff_settings(old_val, new_val, field_path))

        return changes

    # ==================== 变更通知 ====================

    def _notify_changes(self, new_settings):
        changes = self._diff_settings(self._settings, new_settings)
        if not changes:
            return

        print(f"[ConfigManager] Config changed:")
        for path, old_val, new_val in changes:
            print(f"  {path}: {old_val} → {new_val}")

        # 全局回调
        for cb in self._callbacks:
            try:
                cb(changes)
            except Exception as e:
                print(f"[ConfigManager] Global callback error: {e}")

        # 路径级回调
        for path, old_val, new_val in changes:
            for prefix, cbs in self._path_callbacks.items():
                if path == prefix or path.startswith(prefix + "."):
                    for cb in cbs:
                        try:
                            cb(old_val, new_val)
                        except Exception as e:
                            print(f"[ConfigManager] Path callback error ({prefix}): {e}")

    def on_change(self, path: str = ""):
        """
        注册配置变更回调

        Usage:
            @manager.on_change()              # 全局，接收所有变更
            def handler(changes): ...

            @manager.on_change("llm.provider")   # 路径级，当 llm.provider 或其子路径变化时触发
            def handler(old, new): ...
        """
        if path:
            def decorator(cb):
                self._path_callbacks.setdefault(path, []).append(cb)
                return cb
            return decorator
        else:
            def decorator(cb):
                self._callbacks.append(cb)
                return cb
            return decorator


# ==================== 全局单例 ====================

_manager: ConfigManager | None = None

def init_config(default_path: str = "config/default.yaml", env_name: str | None = None) -> ConfigManager:
    global _manager
    _manager = ConfigManager(default_path, env_name)
    return _manager

def get_settings() -> "Settings":
    if _manager is None:
        raise RuntimeError("Config not initialized. Call init_config() first.")
    return _manager.settings

def get_manager() -> ConfigManager:
    if _manager is None:
        raise RuntimeError("Config not initialized. Call init_config() first.")
    return _manager
```

### 6.3 热更新策略矩阵

| 配置项 | 热更新行为 | 说明 |
|--------|-----------|------|
| `llm.provider` | **重建 LLM 实例** | 切换模型/提供商，不影响已有请求，新请求用新实例 |
| `llm.providers.*.model` | **重建 LLM 实例** | 同上 |
| `llm.providers.*.temperature` | **重建 LLM 实例** | 简单参数也走重建（实现成本低、安全） |
| `agent.system_prompt` | **立即生效** | Agent 直接引用 config 引用，无需重启 |
| `agent.context_window` | **新请求生效** | 已有会话不受影响 |
| `agent.max_chain_depth` | **新请求生效** | 已在进行的链式调用按旧值完成 |
| `tools.builtin.*.enabled` | **动态启停** | ToolRegistry.mark_enabled/disabled，无需重建 |
| `tools.builtin.*.settings` | **下次调用生效** | 工具执行时读取最新配置 |
| `session.ttl_seconds` | **新会话生效** | 已有会话按创建时的 TTL |
| `server.port` | **需要重启** | 端口无法热更新（标记为 unsupported） |
| `logging.level` | **立即生效** | 直接调用 logging.getLogger().setLevel() |

### 6.4 Agent 侧响应配置变更

```python
# core/agent.py 中注册回调

@config_manager.on_change("llm")
async def on_llm_config_changed(old, new):
    """LLM 配置变化 → 重建 LLM 实例"""
    from llm.factory import create_llm
    settings = get_settings()
    llm_config = settings.llm.active
    agent.llm = create_llm(settings.llm.provider, llm_config)
    print(f"[Agent] LLM reloaded: {settings.llm.provider}/{llm_config.model}")

@config_manager.on_change("tools")
def on_tools_config_changed(old, new):
    """工具配置变化 → 刷新注册表"""
    from tools.registry import ToolRegistry
    settings = get_settings()
    for name, tool_cfg in settings.tools.builtin.items():
        ToolRegistry.set_enabled(name, tool_cfg.enabled)
    print("[Agent] Tools config refreshed")

@config_manager.on_change("agent.system_prompt")
def on_system_prompt_changed(old, new):
    """System prompt 变化 → 更新上下文构建器"""
    agent.context_builder.system_prompt = new
    print(f"[Agent] System prompt updated")
```

---

## 7. 环境变量模板 `.env.example`

```bash
# LLM API Keys
OPENAI_API_KEY=sk-xxx
ANTHROPIC_API_KEY=sk-ant-xxx
AZURE_API_KEY=xxx
AZURE_ENDPOINT=https://xxx.openai.azure.com/

# Search API Key (if using Google)
SEARCH_API_KEY=xxx

# Redis (when session backend = redis)
REDIS_URL=redis://localhost:6379/0
```

---

## 8. CLI 参数设计

```python
# main.py
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="YellowBull Agent")
    parser.add_argument("--env", type=str, default=None,
                        help="Environment name (loads environments/{name}.yaml)")
    parser.add_argument("--config", type=str, default="config/default.yaml",
                        help="Path to default config file")
    parser.add_argument("--watch-interval", type=float, default=2.0,
                        help="Config file watch poll interval in seconds")
    return parser.parse_args()
```

CLI args → env_name → 加载对应 environment yaml，三层优先级：

```
default.yaml (base)
    ↓ deep_merge
environments/{env}.yaml (override)
    ↓ resolve
${ENV_VAR} from os.environ
```

---

## 9. 设计总结

| 特性 | 实现方式 |
|------|---------|
| **类型安全** | dataclass + IDE 补全，编译期无错误，运行时零反射 |
| **热更新** | 文件轮询（跨平台）+ hash 对比 + diff 通知 |
| **分层覆盖** | deep_merge，环境配置只写差异 |
| **敏感信息** | `${ENV_VAR}` 引用，不入库 |
| **变更粒度** | 支持全局回调和路径级回调 |
| **向前兼容** | 忽略 YAML 中 dataclass 未定义的字段 |
| **跨平台** | 轮询而非 inotify/fsevents，Windows/Linux/macOS 通用 |

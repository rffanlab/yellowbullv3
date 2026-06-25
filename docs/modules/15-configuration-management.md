# 配置管理详细设计

## 1. 设计目标

| 目标 | 说明 |
|------|------|
| **分层覆盖** | defaults.yaml → env-specific → .env 文件，优先级递增 |
| **热重载** | 支持运行时重新加载部分配置（日志级别、限流参数等） |
| **强类型验证** | Pydantic Settings 模型，启动时校验所有必填字段 |
| **敏感信息保护** | API Key 等通过环境变量注入，不写入配置文件 |

---

## 2. 配置层次结构 `config/settings.py`

```python
"""
YellowBull 配置管理系统。

优先级（低 → 高）：
1. config/defaults.yaml        — 默认值
2. config/{environment}.yaml   — 环境覆盖（dev/staging/prod）
3. .env / 环境变量             — 敏感信息、运行时覆盖
4. CLI arguments               — 启动参数

Usage:
    settings = get_settings()          # 获取全局配置单例
    llm_config = settings.llm         # LLM 相关配置
    db_url = settings.database.url    # 数据库连接串
"""

from pathlib import Path
from typing import Optional
import yaml


# ---------- Pydantic Settings Models ----------

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator


class LLMProviderConfig(BaseSettings):
    """LLM Provider 配置"""
    api_key: str = Field(..., description="API Key（从环境变量读取）")
    base_url: Optional[str] = None
    model_name: str = "gpt-4"
    max_tokens: int = 4096
    temperature: float = 0.7
    timeout_seconds: float = 60.0

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v):
        if not 0.0 <= v <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        return v


class DatabaseConfig(BaseSettings):
    """数据库配置"""
    url: str = Field(..., description="Database connection URL")
    pool_size: int = 5
    max_overflow: int = 10
    echo: bool = False

    @field_validator("url")
    @classmethod
    def validate_url(cls, v):
        if not v or "://" not in v:
            raise ValueError("Invalid database URL")
        return v


class RAGConfig(BaseSettings):
    """RAG Pipeline 配置"""
    embedding_model: str = "text-embedding-3-small"
    top_k: int = 5
    similarity_threshold: float = 0.7
    chunk_size: int = 1000
    chunk_overlap: int = 200


class MemoryConfig(BaseSettings):
    """记忆系统配置"""
    max_short_term_messages: int = 50
    max_long_term_entries: int = 1000
    summary_trigger_interval: int = 10       # 每 N 条消息触发一次摘要
    ttl_seconds: Optional[int] = None        # None = 永不过期


class SecurityConfig(BaseSettings):
    """安全配置"""
    jwt_secret_key: str = Field(..., alias="JWT_SECRET_KEY")
    jwt_algorithm: str = "HS256"
    access_token_minutes: int = 15
    refresh_token_days: int = 7

    rate_limit_global: int = 1000
    rate_limit_user: int = 60
    rate_limit_session: int = 20


class WebSocketConfig(BaseSettings):
    """WebSocket 配置"""
    max_connections: int = 100
    heartbeat_interval_seconds: float = 30.0
    message_timeout_seconds: float = 120.0
    ping_interval_seconds: float = 25.0


class LoggingConfig(BaseSettings):
    """日志配置（支持热重载）"""
    level: str = "INFO"
    format: str = "json"          # json / text
    file_path: Optional[str] = None
    max_file_size_mb: int = 100
    backup_count: int = 5


class AppSettings(BaseSettings):
    """应用主配置（聚合所有子配置）"""

    model_config = SettingsConfigDict(
        env_prefix="YELLOWBULL_",
        env_nested_delimiter="__",   # YELLOWBULL_LLM__API_KEY
        extra="ignore",
    )

    # 环境标识
    environment: str = Field(default="development", description="dev/staging/prod")
    debug: bool = False

    # 子配置
    llm: LLMProviderConfig = Field(default_factory=LLMProviderConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    websocket: WebSocketConfig = Field(default_factory=WebSocketConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v):
        if v not in ("development", "staging", "production"):
            raise ValueError(f"Invalid environment: {v}")
        return v


# ---------- YAML Config Loader ----------

class YamlConfigLoader:
    """YAML 配置文件加载器"""

    def __init__(self, config_dir: str = "config"):
        self._config_dir = Path(config_dir)

    def load_merged(self, environment: str = "development") -> dict:
        """
        合并加载配置：defaults.yaml → {environment}.yaml

        Returns:
            合并后的配置字典（供 Pydantic Settings 使用）
        """
        merged: dict = {}

        # 1. defaults.yaml
        defaults_path = self._config_dir / "defaults.yaml"
        if defaults_path.exists():
            with open(defaults_path) as f:
                merged.update(yaml.safe_load(f) or {})

        # 2. environment-specific override
        env_path = self._config_dir / f"{environment}.yaml"
        if env_path.exists():
            with open(env_path) as f:
                merged.update(yaml.safe_load(f) or {})

        return merged


# ---------- Hot Reload Support ----------

import asyncio
from typing import Callable


class ConfigHotReload:
    """
    配置热重载管理器。

    支持热重载的配置项：
    - logging.level          → 日志级别
    - security.rate_limit_*  → 限流参数
    - websocket.*            → WebSocket 参数

    触发方式：
    - SIGHUP 信号（Unix）
    - HTTP API: POST /api/admin/config/reload
    - 文件监听（开发环境）
    """

    def __init__(self, settings: AppSettings):
        self._settings = settings
        self._reload_callbacks: list[Callable] = []

    def on_reload(self, callback: Callable):
        """注册热重载回调"""
        self._reload_callbacks.append(callback)

    async def reload_logging(self, new_level: str):
        """热重载日志级别"""
        import logging
        self._settings.logging.level = new_level
        root_logger = logging.getLogger()
        root_logger.setLevel(new_level.upper())
        for handler in root_logger.handlers:
            handler.setLevel(new_level.upper())

        await self._notify_reload("logging", {"level": new_level})

    async def reload_rate_limits(self, **kwargs):
        """热重载限流参数"""
        for key, value in kwargs.items():
            if hasattr(self._settings.security, key):
                setattr(self._settings.security, key, value)

        await self._notify_reload("security", kwargs)

    async def _notify_reload(self, section: str, changes: dict):
        """通知所有注册的回调"""
        for cb in self._reload_callbacks:
            try:
                await cb(section, changes)
            except Exception as e:
                import logging
                logging.getLogger(__name__).error(
                    f"Config reload callback failed: {e}"
                )


# ---------- Global Singleton ----------

_settings_instance: Optional[AppSettings] = None
_hot_reload_instance: Optional[ConfigHotReload] = None


def get_settings() -> AppSettings:
    """获取全局配置单例"""
    global _settings_instance
    if _settings_instance is None:
        raise RuntimeError("Settings not initialized. Call init_settings() first.")
    return _settings_instance


async def init_settings(environment: str | None = None):
    """初始化配置（应用启动时调用一次）"""
    global _settings_instance, _hot_reload_instance

    import os
    env = environment or os.getenv("YELLOWBULL_ENVIRONMENT", "development")

    # 加载 YAML 配置
    loader = YamlConfigLoader()
    yaml_config = loader.load_merged(env)

    # Pydantic Settings 会自动从环境变量读取，YAML 作为默认值
    _settings_instance = AppSettings(
        environment=env,
        **yaml_config,
    )

    _hot_reload_instance = ConfigHotReload(_settings_instance)

    import logging
    logging.info(f"Settings initialized for environment: {env}")


def get_hot_reload() -> ConfigHotReload:
    """获取热重载管理器"""
    global _hot_reload_instance
    if _hot_reload_instance is None:
        raise RuntimeError("Hot reload not initialized. Call init_settings() first.")
    return _hot_reload_instance
```

---

## 3. YAML 配置文件示例 `config/defaults.yaml`

```yaml
# config/defaults.yaml — 默认配置（所有环境共享）

llm:
  model_name: "gpt-4"
  max_tokens: 4096
  temperature: 0.7
  timeout_seconds: 60

database:
  pool_size: 5
  max_overflow: 10
  echo: false

rag:
  embedding_model: "text-embedding-3-small"
  top_k: 5
  similarity_threshold: 0.7
  chunk_size: 1000
  chunk_overlap: 200

memory:
  max_short_term_messages: 50
  max_long_term_entries: 1000
  summary_trigger_interval: 10

security:
  jwt_algorithm: "HS256"
  access_token_minutes: 15
  refresh_token_days: 7
  rate_limit_global: 1000
  rate_limit_user: 60
  rate_limit_session: 20

websocket:
  max_connections: 100
  heartbeat_interval_seconds: 30
  message_timeout_seconds: 120
  ping_interval_seconds: 25

logging:
  level: "INFO"
  format: "json"
```

`config/production.yaml`:
```yaml
# config/production.yaml — 生产环境覆盖

debug: false

llm:
  max_tokens: 8192
  timeout_seconds: 120

database:
  pool_size: 20
  max_overflow: 40
  echo: false

security:
  rate_limit_global: 5000
  rate_limit_user: 120

logging:
  level: "WARNING"
```

---

## 4. Admin API 配置管理 `api/admin/config.py`

```python
"""Admin API：运行时配置管理"""

from fastapi import APIRouter, Depends, HTTPException, status

router = APIRouter(prefix="/api/admin", tags=["admin"])


@router.post("/config/reload")
async def reload_config(
    section: str,
    payload: dict,
    admin_user = Depends(require_admin),  # 仅管理员可操作
):
    """热重载指定配置段"""
    hot_reload = get_hot_reload()

    if section == "logging":
        new_level = payload.get("level")
        if not new_level:
            raise HTTPException(400, "Missing 'level' field")
        await hot_reload.reload_logging(new_level)

    elif section == "security":
        await hot_reload.reload_rate_limits(**payload)

    else:
        raise HTTPException(400, f"Unsupported config section: {section}")

    return {"status": "reloaded", "section": section}


@router.get("/config")
async def get_current_config(
    admin_user = Depends(require_admin),
):
    """查看当前配置（脱敏）"""
    settings = get_settings()
    # 移除敏感字段
    config_dict = settings.model_dump()
    if "api_key" in str(config_dict.get("llm", {})):
        config_dict["llm"]["api_key"] = "***REDACTED***"
    if "jwt_secret_key" in str(config_dict.get("security", {})):
        config_dict["security"]["jwt_secret_key"] = "***REDACTED***"
    return config_dict
```

---

## 5. 架构总览

```
                    ┌─────────────────────┐
                    │   AppSettings       │ ← Pydantic Settings 主模型
                    │   (强类型 + 验证)    │
                    └──────────┬──────────┘
                               │ composed of
          ┌────────────────────┼────────────────────┐
          ▼                    ▼                     ▼
   ┌─────────────┐    ┌─────────────┐      ┌─────────────┐
   │ LLMProvider │    │ Database    │      │ Security    │
   │ RAG         │    │ Memory      │      │ WebSocket   │
   │ Logging     │    └─────────────┘      │ Logging     │
   └─────────────┘                         └─────────────┘

                    ┌─────────────────────┐
                    │  Config Loading     │ ← 优先级递增
                    ├─────────────────────┤
                    │ 1. defaults.yaml    │
                    │ 2. {env}.yaml       │
                    │ 3. .env / env vars  │
                    │ 4. CLI arguments    │
                    └─────────────────────┘

                    ┌─────────────────────┐
                    │  Hot Reload         │ ← 运行时更新
                    ├─────────────────────┤
                    │ SIGHUP signal       │
                    │ Admin API endpoint  │
                    │ File watcher (dev)  │
                    └─────────────────────┘
```

---

## 6. 设计总结

| 特性 | 实现方式 |
|------|---------|
| **分层覆盖** | YAML defaults → env-specific → .env，优先级递增 |
| **强类型验证** | Pydantic Settings + field_validator，启动时校验 |
| **环境变量注入** | `YELLOWBULL_LLM__API_KEY` 嵌套命名约定 |
| **热重载** | logging / security / websocket 支持运行时更新 |
| **Admin API** | POST `/api/admin/config/reload`，仅管理员可操作 |
| **敏感信息保护** | API Key / JWT Secret 通过环境变量注入，配置输出脱敏 |

# Logging Setup 模块详细设计

## 1. 概述

Logging Setup 提供 JSON 结构化日志系统，支持控制台输出和文件轮转。所有模块通过 `get_logger` 获取子日志器进行日志记录。

**对应源码:** `core/logging_setup.py`

### 职责
- JSON 格式日志输出，便于机器解析和日志聚合
- 文件轮转机制，防止日志文件无限增长
- 结构化上下文字段支持（session_id, user_id）
- 第三方库日志降噪

## 2. API 设计

### JsonFormatter

```python
class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str
```

将 Python LogRecord 格式化为 JSON 字符串。

**输出字段:**

| 字段 | 类型 | 说明 |
|---|---|---|
| `timestamp` | string | UTC ISO8601 时间戳，如 `"2025-01-01T12:00:00Z"` |
| `level` | string | 日志级别：DEBUG, INFO, WARNING, ERROR, CRITICAL |
| `logger` | string | 日志器名称（模块名） |
| `message` | string | 日志消息内容 |
| `session_id` | string (optional) | 会话 ID，通过 `extra` 参数传入 |
| `user_id` | string (optional) | 用户 ID，通过 `extra` 参数传入 |
| `exception` | string (optional) | 异常堆栈信息（仅错误日志） |

**示例输出:**
```json
{"timestamp": "2025-01-01T12:00:00Z", "level": "INFO", "logger": "core.agent", "message": "Processing request", "session_id": "abc-123", "user_id": "user-1"}
```

### setup_logging

```python
def setup_logging(level: str = "INFO") -> logging.Logger
```

配置根日志器，设置日志级别和处理器。

**参数:**

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `level` | string | `"INFO"` | 日志级别字符串 |

**行为:**
1. 设置根日志器级别
2. 添加控制台处理器（JSON 格式，输出到 stdout）
3. 添加文件轮转处理器：
   - 目录: `logs/`
   - 文件名: `agent.log`
   - 单文件大小上限: 50MB
   - 保留副本数: 5
4. 抑制第三方库日志（httpx, openai, anthropic, uvicorn.access → WARNING）

### get_logger

```python
def get_logger(name: str) -> logging.Logger
```

获取命名子日志器。各模块通过 `get_logger(__name__)` 获取对应名称的日志器。

## 3. 使用方式

### 基本用法
```python
from core.logging_setup import get_logger

logger = get_logger(__name__)
logger.info("Simple log message")
```

### 带结构化上下文
```python
logger.info("Processing request", extra={
    "session_id": session.session_id,
    "user_id": request.user_id,
})
```

### 异常日志
```python
try:
    ...
except Exception as e:
    logger.error("Operation failed", exc_info=True)
```

## 4. 与主文档的对应关系

| agent-design.md 章节 | 本模块覆盖内容 |
|---|---|
| 日志与监控 - JSON 结构化日志 | ✅ JsonFormatter 实现 |
| 日志与监控 - 文件轮转 | ✅ RotatingFileHandler, 50MB × 5 |
| 日志与监控 - 会话追踪字段 | ✅ session_id, user_id 通过 extra 传入 |

## 5. 依赖关系

```
logging_setup
    ├── logging (Python stdlib)
    ├── logging.handlers.RotatingFileHandler
    └── json, sys, datetime, pathlib
```

## 6. 注意事项

- 时间戳使用 `datetime.utcnow()`，始终为 UTC 时区
- JSON 输出使用 `ensure_ascii=False`，支持中文等 Unicode 字符
- 文件轮转在达到 50MB 时触发，保留最近 5 个副本（总计约 250MB）
- 生产环境建议接入 ELK/Loki 等日志聚合系统

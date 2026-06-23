# API Layer 模块详细设计

## 1. 概述

API Layer 基于 FastAPI 框架提供 RESTful HTTP 接口，是系统与外部交互的唯一入口。包含聊天、会话管理和健康检查等端点。

**对应源码:** `api/server.py`

### 职责
- HTTP 请求接收与响应
- CORS 跨域支持
- Pydantic 请求/响应模型验证
- Agent 实例生命周期管理

## 2. API 端点设计

### POST /api/chat

发送聊天消息，触发 Agent 处理流程。

**Request Body:**
```json
{
    "session_id": "optional-session-id",
    "user_id": "user-123",
    "message": "你好"
}
```

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `session_id` | string | 否 | 会话 ID；为空则创建新会话 |
| `user_id` | string | 是 | 用户唯一标识 |
| `message` | string | 是 | 用户消息内容 |

**Response:**
```json
{
    "content": "回答内容",
    "session_id": "generated-or-existing-id",
    "tool_results": [],
    "needs_clarification": false,
    "usage": {
        "prompt_tokens": 50,
        "completion_tokens": 100,
        "total_tokens": 150
    }
}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `content` | string | Agent 回复内容 |
| `session_id` | string | 会话 ID（新建或复用） |
| `tool_results` | array | 工具执行结果列表（预留） |
| `needs_clarification` | bool | 是否需要用户澄清 |
| `usage` | object (nullable) | Token 用量统计 |

### DELETE /api/sessions/{session_id}

删除指定会话。

**Response:**
- `204 No Content`: 删除成功
- `404 Not Found`: 会话不存在

### GET /api/sessions/{session_id}/history

获取会话历史消息。

**Response:**
```json
{
    "session_id": "abc-123",
    "messages": [
        {
            "role": "user",
            "content": "你好",
            "created_at": "2025-01-01 12:00:00"
        },
        {
            "role": "assistant",
            "content": "你好！有什么可以帮助你的？",
            "created_at": "2025-01-01 12:00:01"
        }
    ]
}
```

**Response:**
- `404 Not Found`: 会话不存在
- `500 Internal Server Error`: Agent 未初始化

### GET /api/health

健康检查端点。

**Response:**
```json
{
    "status": "ok"
}
```

## 3. 应用工厂模式

### create_app

```python
def create_app() -> FastAPI
```

应用工厂函数，用于 `uvicorn --factory` 模式启动。

**初始化流程:**
1. 创建 FastAPI 实例
2. 添加 CORS 中间件（允许所有来源、方法、头）
3. 加载配置 (`load_settings`)
4. 根据配置创建 LLM 实例 (`create_llm`)
5. 创建 Agent 实例并全局保存

## 4. 数据模型

### ChatRequest (Pydantic)

```python
class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    user_id: str
    message: str
```

### ChatResponse (Pydantic)

```python
class ChatResponse(BaseModel):
    content: str
    session_id: str
    tool_results: list[dict] = []
    needs_clarification: bool = False
    usage: Optional[dict] = None
```

## 5. 与主文档的对应关系

| agent-design.md 章节 | 本模块覆盖内容 |
|---|---|
| API层设计 - RESTful 端点 | ✅ /api/chat, /api/sessions, /api/health |
| API层设计 - FastAPI + Pydantic | ✅ 请求/响应模型验证 |
| API层设计 - CORS 支持 | ✅ CORSMiddleware |

## 6. 依赖关系

```
api/server
    ├── fastapi (Web 框架)
    ├── pydantic (数据验证)
    ├── config.settings.load_settings
    ├── llm.factory.create_llm
    └── core.agent.Agent, ChatRequest
```

## 7. 注意事项

- Agent 实例为全局单例，通过 `_agent` 模块变量持有
- CORS 当前允许所有来源，生产环境应限制具体域名
- `/api/chat` 端点为同步 HTTP 接口，Agent 内部异步执行（FastAPI 自动处理）
- `tool_results` 字段目前返回空列表，预留用于未来扩展

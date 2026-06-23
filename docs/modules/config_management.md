# Configuration Management 模块详细设计

## 1. 概述

配置管理模块提供 YAML 配置文件加载、环境变量解析和类型安全的配置模型。支持多环境配置覆盖，是系统启动时第一个初始化的组件。

**对应源码:** `config/settings.py`

### 职责
- YAML 配置文件加载与解析
- `${ENV_VAR}` 格式的环境变量替换
- Pydantic 数据验证与默认值填充
- 多 Provider LLM 配置管理

## 2. 类设计

### Settings (根配置模型)

```python
@dataclass
class Settings:
    llm: LLMConfig
    agent: AgentConfig
    server: ServerConfig
```

聚合所有子配置模块。

### LLMConfig

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `active` | string | `"openai"` | 当前激活的 Provider 名称 |
| `providers` | dict[str, dict] | `{}` | 各 Provider 配置映射，key 为 provider 名 |

**YAML 示例:**
```yaml
llm:
  active: openai
  providers:
    openai:
      api_key: "${OPENAI_API_KEY}"
      model: gpt-4o-mini
      max_tokens: 8192
      temperature: 0.7
    anthropic:
      api_key: "${ANTHROPIC_API_KEY}"
      model: claude-sonnet-4-20250514
```

### AgentConfig

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `system_prompt` | string | `"你是一个智能助手..."` | Agent 系统提示词 |
| `max_chain_depth` | int | `5` | ReAct 链式调用最大深度 |
| `context_window` | int | `48` | 上下文滑动窗口大小（消息条数） |
| `max_tool_calls_per_turn` | int | `10` | 每轮最大并行工具调用数 |
| `tool_retry_limit` | int | `2` | 单个工具失败重试次数 |

### ServerConfig

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `host` | string | `"0.0.0.0"` | 服务监听地址 |
| `port` | int | `8000` | 服务端口 |
| `debug` | bool | `False` | 调试模式开关 |

## 3. API 设计

### load_settings

```python
def load_settings(config_path: str = "config/default.yaml") -> Settings
```

加载配置文件并返回 Settings 实例。

**处理流程:**
1. 检查文件是否存在，不存在则使用默认配置
2. 读取 YAML 内容
3. 递归解析 `${ENV_VAR}` 格式的环境变量引用
4. 按字段映射到 Pydantic 模型，缺失字段使用默认值

### _resolve_env_vars (内部函数)

```python
def _resolve_env_vars(value: Any) -> Any
```

递归解析环境变量引用。

**行为:**
- 字符串: 查找 `${VAR_NAME}` 模式，替换为对应环境变量值
- 字典: 递归处理每个值
- 列表: 递归处理每个元素
- 其他类型: 直接返回

## 4. 配置加载优先级

```
默认值 (Pydantic field default)
    ↑ 被覆盖
YAML 配置文件
    ↑ 被覆盖
环境变量 (${VAR} 引用)
```

最终生效值 = 环境变量 > YAML 配置 > Pydantic 默认值

## 5. 与主文档的对应关系

| agent-design.md 章节 | 本模块覆盖内容 |
|---|---|
| 配置管理 - YAML 配置文件 | ✅ load_settings 加载 YAML |
| 配置管理 - 环境变量覆盖 | ✅ _resolve_env_vars 解析 ${VAR} |
| 配置管理 - Pydantic 验证 | ✅ Settings/LLMConfig/AgentConfig/ServerConfig |

## 6. 依赖关系

```
config/settings
    ├── pydantic (数据验证)
    └── yaml (PyYAML, 文件解析)
```

## 7. 注意事项

- API Key 等敏感信息必须通过环境变量注入，禁止明文写入 YAML
- 配置文件缺失不会导致启动失败（使用默认配置）
- Provider 名称需与 `llm/factory.py` 中的注册名一致

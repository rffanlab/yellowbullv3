# 配置系统 — 验收标准

## CFG-01: YAML 配置文件加载

**优先级**: P0

- **Given** `config/settings.yaml` 存在且格式正确
- **When** 调用 `ConfigManager.load()`
- **Then** 所有顶层 key（app、llm、session、tools、rag）被解析为对应的 Settings dataclass
- **验证方式**: 单元测试 — 断言各字段值与 YAML 一致

## CFG-02: 环境变量覆盖

**优先级**: P0

- **Given** YAML 中 `api_key` 设置为占位符，环境变量 `OPENAI_API_KEY=sk-test` 已设置
- **When** 加载配置
- **Then** `${ENV_VAR}` 语法被替换为对应环境变量值；未设置的环境变量保留原值并记录 warning
- **验证方式**: 单元测试 — mock 环境变量后断言替换结果

## CFG-03: 配置热重载

**优先级**: P1

- **Given** 应用已启动，配置文件被外部修改
- **When** `ConfigManager.reload()` 被调用（或 file watcher 触发）
- **Then** 新配置生效，不重启进程；旧配置引用不受影响
- **验证方式**: 集成测试 — 修改 YAML 后调用 reload()，断言字段值更新

## CFG-04: 默认值填充

**优先级**: P0

- **Given** `settings.yaml` 缺失部分可选字段（如 `session.max_turns`）
- **When** 加载配置
- **Then** 缺失字段使用 Settings dataclass 的默认值填充，不报错
- **验证方式**: 单元测试 — 最小 YAML 文件加载后断言默认值

## CFG-05: 配置校验

**优先级**: P1

- **Given** `settings.yaml` 中 `llm.provider` 设置为未知值（如 "unknown"）
- **When** 加载配置
- **Then** 抛出 `ConfigValidationError`，包含具体字段和原因
- **验证方式**: 单元测试 — 断言异常类型和错误消息

## CFG-06: 多环境配置

**优先级**: P1

- **Given** 存在 `settings.dev.yaml` 和 `settings.prod.yaml`
- **When** 通过环境变量 `APP_ENV=prod` 指定环境
- **Then** 优先加载对应环境的配置文件，与基础配置合并
- **验证方式**: 集成测试 — 切换 APP_ENV 后断言配置差异

## CFG-07: 敏感信息保护

**优先级**: P0

- **Given** 配置包含 `api_key` 等敏感字段
- **When** 打印或日志输出 ConfigManager 对象
- **Then** 敏感字段被脱敏显示（如 `sk-***`）
- **验证方式**: 单元测试 — 断言 `__repr__` / `__str__` 输出

## CFG-08: 配置 Schema 版本管理

**优先级**: P2

- **Given** YAML 中 `version` 字段与代码期望版本不一致
- **When** 加载配置
- **Then** 记录 warning，尝试自动迁移或提示用户更新
- **验证方式**: 单元测试 — mock 不同版本号断言行为

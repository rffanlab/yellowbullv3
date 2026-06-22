# 前端 UI — 验收标准

## UI-01: Chat Interface — 消息发送与显示

**优先级**: P0

- **Given** 用户访问 `/`（聊天页面）
- **When** 输入消息并点击发送
- **Then** 用户消息立即显示在对话区；AI 回复以流式方式逐字显示；消息按时间顺序排列
- **验证方式**: E2E 测试（Playwright）

## UI-02: Chat Interface — SSE 流式接收

**优先级**: P0

- **Given** 聊天页面已加载
- **When** AI 回复通过 SSE 返回
- **Then** 前端逐 chunk 更新 UI；不阻塞主线程；打字动画流畅（>= 30fps）
- **验证方式**: E2E 测试

## UI-03: Chat Interface — 代码块渲染

**优先级**: P1

- **Given** AI 回复包含 Markdown 代码块（```python ... ```）
- **When** 消息渲染完成
- **Then** 代码块使用语法高亮显示；包含复制按钮；支持折叠/展开
- **验证方式**: E2E 测试

## UI-04: Chat Interface — 工具调用可视化

**优先级**: P1

- **Given** AgentCore 执行了工具调用
- **When** SSE stream 返回工具结果
- **Then** 显示 "正在搜索..." / "正在执行代码..." 等状态提示；完成后收起或保留结果摘要
- **验证方式**: E2E 测试

## UI-05: Session Management — 会话列表

**优先级**: P1

- **Given** 用户有多个历史 sessions
- **When** 打开侧边栏
- **Then** 显示 session 列表（标题、时间）；点击切换对话；支持新建/删除/重命名
- **验证方式**: E2E 测试

## UI-06: Session Management — 本地持久化

**优先级**: P1

- **Given** 浏览器 localStorage
- **When** 页面刷新
- **Then** 当前 session_id 和消息列表从本地存储恢复；与服务端状态同步
- **验证方式**: E2E 测试

## UI-07: WebSocket Fallback

**优先级**: P1

- **Given** 后端支持 WebSocket
- **When** SSE 连接失败（如代理不支持）
- **Then** 自动降级为 WebSocket；用户无感知切换
- **验证方式**: E2E 测试 — mock SSE failure

## UI-08: Error Handling — API 错误展示

**优先级**: P1

- **Given** API 返回 5xx 或超时
- **When** 请求失败
- **Then** 显示友好错误提示（"服务暂时不可用，请稍后重试"）；提供重试按钮；不暴露技术细节
- **验证方式**: E2E 测试 — mock API error

## UI-09: Error Handling — 网络断开处理

**优先级**: P1

- **Given** SSE/WebSocket 连接中
- **When** 网络断开后恢复
- **Then** 自动重连（最多 3 次）；重连失败显示提示；不丢失已发送的消息
- **验证方式**: E2E 测试 — network throttling

## UI-10: Responsive Design

**优先级**: P2

- **Given** 不同屏幕尺寸（手机、平板、桌面）
- **When** 访问聊天页面
- **Then** 布局自适应；侧边栏在小屏下可折叠；输入框和按钮可点击
- **验证方式**: E2E 测试 — viewport testing

## UI-11: Theme — 暗色/亮色模式

**优先级**: P2

- **Given** 系统主题设置
- **When** 页面加载 / 用户切换主题
- **Then** 跟随系统或手动选择；主题偏好保存到 localStorage；切换无闪烁
- **验证方式**: E2E 测试

## UI-12: Accessibility — 基础无障碍

**优先级**: P2

- **Given** 聊天界面
- **When** 使用键盘导航（Tab/Enter）
- **Then** 所有交互元素可聚焦；消息区域有 ARIA label；颜色对比度符合 WCAG AA
- **验证方式**: 手动检查 + axe-core

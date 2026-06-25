# 安全与权限详细设计

## 1. 设计目标

| 目标 | 说明 |
|------|------|
| **RBAC 三级权限** | public / authenticated / admin，覆盖所有 API 端点 |
| **JWT 认证** | access_token (短期) + refresh_token (长期)，支持无状态验证 |
| **多级限流** | 全局 → 用户级 → 会话级 → 端点级，四级限流策略 |
| **输入校验** | 所有外部输入经过 schema 验证，防止注入攻击 |
| **审计日志** | 关键操作记录完整审计轨迹，支持合规审查 |

---

## 2. RBAC 权限模型 `security/rbac.py`

```python
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Callable, Awaitable, Any
import functools


class Role(str, Enum):
    """三级角色"""
    PUBLIC = "public"          # 未认证用户，仅可访问公开端点
    AUTHENTICATED = "authenticated"  # 已登录用户，可创建会话、使用工具
    ADMIN = "admin"            # 管理员，全部权限


class Resource(str, Enum):
    """受保护资源"""
    SESSIONS = "sessions"
    MESSAGES = "messages"
    TOOLS = "tools"
    RAG_KNOWLEDGE = "rag_knowledge"
    MEMORY = "memory"
    PLUGINS = "plugins"
    SETTINGS = "settings"
    AUDIT_LOGS = "audit_logs"


class Action(str, Enum):
    """操作类型"""
    CREATE = auto()
    READ = auto()
    UPDATE = auto()
    DELETE = auto()
    EXECUTE = auto()  # 工具执行、插件调用等特殊操作


@dataclass(frozen=True)
class Permission:
    """权限规则：角色 + 资源 + 操作"""
    role: Role
    resource: Resource
    actions: set[Action]

    def allows(self, resource: Resource, action: Action) -> bool:
        return resource == self.resource and action in self.actions


# ---------- 默认权限矩阵 ----------

DEFAULT_PERMISSIONS: list[Permission] = [
    # PUBLIC：仅健康检查、API文档等公开端点
    Permission(Role.PUBLIC, Resource.TOOLS, {Action.READ}),

    # AUTHENTICATED：核心功能
    Permission(Role.AUTHENTICATED, Resource.SESSIONS,
               {Action.CREATE, Action.READ, Action.UPDATE, Action.DELETE}),
    Permission(Role.AUTHENTICATED, Resource.MESSAGES,
               {Action.CREATE, Action.READ}),
    Permission(Role.AUTHENTICATED, Resource.TOOLS,
               {Action.READ, Action.EXECUTE}),
    Permission(Role.AUTHENTICATED, Resource.RAG_KNOWLEDGE,
               {Action.READ}),
    Permission(Role.AUTHENTICATED, Resource.MEMORY,
               {Action.CREATE, Action.READ, Action.DELETE}),

    # ADMIN：全部权限
]


class PermissionMatrix:
    """
    权限矩阵管理器。

    职责：
    - 维护角色 → 资源 → 操作的映射关系
    - 支持运行时动态添加/移除权限规则
    - 提供 O(1) 的权限检查接口
    """

    def __init__(self, permissions: list[Permission] | None = None):
        # role -> resource -> set[action]
        self._matrix: dict[Role, dict[Resource, set[Action]]] = {
            role: {} for role in Role
        }
        for perm in (permissions or DEFAULT_PERMISSIONS):
            self.grant(perm.role, perm.resource, perm.actions)

    def grant(self, role: Role, resource: Resource, actions: set[Action]):
        """授予角色对资源的操作权限"""
        if resource not in self._matrix[role]:
            self._matrix[role][resource] = set()
        self._matrix[role][resource].update(actions)

    def revoke(self, role: Role, resource: Resource, actions: set[Action]):
        """撤销角色对资源的操作权限"""
        if resource in self._matrix[role]:
            self._matrix[role][resource] -= actions
            if not self._matrix[role][resource]:
                del self._matrix[role][resource]

    def check(self, role: Role, resource: Resource, action: Action) -> bool:
        """检查角色是否有权限执行操作"""
        # Admin 拥有所有权限
        if role == Role.ADMIN:
            return True

        resource_actions = self._matrix.get(role, {}).get(resource, set())
        return action in resource_actions


# ---------- FastAPI 装饰器 ----------

def require_role(*roles: Role):
    """FastAPI 依赖注入：检查用户角色"""
    async def _check(user_role: Role = functools.partial(get_user_role)):
        if user_role not in roles and Role.ADMIN not in roles:
            from fastapi import HTTPException, status
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Required role: {', '.join(r.value for r in roles)}",
            )
        return user_role
    return _check


def require_permission(resource: Resource, action: Action):
    """FastAPI 依赖注入：检查资源操作权限"""
    async def _check(
        user_role: Role = functools.partial(get_user_role),
        matrix: PermissionMatrix = functools.partial(get_permission_matrix),
    ):
        if not matrix.check(user_role, resource, action):
            from fastapi import HTTPException, status
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission denied: {user_role.value} cannot "
                       f"{action.name} on {resource.value}",
            )
        return user_role
    return _check


def get_user_role() -> Role:
    """从请求上下文获取用户角色（由 auth middleware 注入）"""
    ...

def get_permission_matrix() -> PermissionMatrix:
    """获取全局权限矩阵实例"""
    ...
```

---

## 3. JWT 认证 `security/auth.py`

```python
import time
import secrets
from dataclasses import dataclass, field
from typing import Optional
from datetime import timedelta


@dataclass(frozen=True)
class JWTPayload:
    """JWT Token 载荷"""
    sub: str                          # user_id
    role: str                         # 角色标识
    iat: float                        # issued at
    exp: float                        # expiration time
    jti: str = ""                     # JWT ID（用于撤销）


class JWTManager:
    """
    JWT Token 管理器。

    双 Token 策略：
    - access_token: 短期有效（默认 15 分钟），随请求传递
    - refresh_token: 长期有效（默认 7 天），仅用于刷新 access_token

    算法：HS256（对称密钥，适合单体部署）
          RS256（非对称密钥，适合微服务架构）
    """

    def __init__(
        self,
        secret_key: str,
        algorithm: str = "HS256",
        access_token_minutes: int = 15,
        refresh_token_days: int = 7,
    ):
        import jwt
        self._jwt = jwt
        self._secret = secret_key
        self._algorithm = algorithm
        self._access_delta = timedelta(minutes=access_token_minutes)
        self._refresh_delta = timedelta(days=refresh_token_days)

        # 已撤销的 JWT ID（内存中，生产环境应使用 Redis）
        self._revoked_jtis: set[str] = set()

    def create_access_token(self, user_id: str, role: str) -> tuple[str, JWTPayload]:
        """创建 access token"""
        now = time.time()
        payload = JWTPayload(
            sub=user_id,
            role=role,
            iat=now,
            exp=now + self._access_delta.total_seconds(),
            jti=secrets.token_hex(16),
        )
        token = self._jwt.encode(
            payload.__dict__,
            self._secret,
            algorithm=self._algorithm,
        )
        return str(token), payload

    def create_refresh_token(self, user_id: str, role: str) -> tuple[str, JWTPayload]:
        """创建 refresh token"""
        now = time.time()
        payload = JWTPayload(
            sub=user_id,
            role=role,
            iat=now,
            exp=now + self._refresh_delta.total_seconds(),
            jti=secrets.token_hex(16),
        )
        token = self._jwt.encode(
            payload.__dict__,
            self._secret,
            algorithm=self._algorithm,
        )
        return str(token), payload

    def verify_token(self, token: str) -> Optional[JWTPayload]:
        """验证并解析 JWT Token"""
        try:
            data = self._jwt.decode(
                token,
                self._secret,
                algorithms=[self._algorithm],
            )
            payload = JWTPayload(**data)

            if payload.jti in self._revoked_jtis:
                return None  # Token 已被撤销

            return payload
        except (self._jwt.ExpiredSignatureError,
                self._jwt.InvalidTokenError,
                KeyError):
            return None

    def revoke_token(self, jti: str):
        """撤销指定 JWT"""
        self._revoked_jtis.add(jti)


# ---------- FastAPI 集成 ----------

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security_scheme = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_scheme),
    jwt_manager: JWTManager = functools.partial(get_jwt_manager),
) -> Optional[JWTPayload]:
    """
    FastAPI 依赖：从 Authorization header 提取并验证 JWT。

    返回 None 表示未认证（公开端点可正常处理）。
    抛出 HTTPException 表示认证失败（需要认证的端点会拦截）。
    """
    if credentials is None:
        return None

    payload = jwt_manager.verify_token(credentials.credentials)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return payload


async def require_auth(
    user: JWTPayload = Depends(get_current_user),
) -> JWTPayload:
    """FastAPI 依赖：要求用户已认证"""
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


async def require_admin(
    user: JWTPayload = Depends(require_auth),
) -> JWTPayload:
    """FastAPI 依赖：要求管理员权限"""
    if user.role != Role.ADMIN.value:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return user
```

---

## 4. 多级限流 `security/rate_limit.py`

```python
import time
import asyncio
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict


@dataclass
class RateLimitRule:
    """限流规则"""
    max_requests: int          # 最大请求数
    window_seconds: float      # 时间窗口（秒）
    burst_allowed: bool = True  # 是否允许突发流量


@dataclass
class RateLimitState:
    """限流状态"""
    timestamps: list[float] = field(default_factory=list)

    def add_request(self, now: float, window: float) -> bool:
        """
        记录请求并检查是否超限。

        Returns:
            True 如果允许通过，False 如果被限流
        """
        # 清理过期时间戳
        cutoff = now - window
        self.timestamps = [t for t in self.timestamps if t > cutoff]
        return len(self.timestamps) < self.max_requests


class TokenBucket:
    """
    令牌桶算法实现。

    相比固定窗口，令牌桶允许突发流量且边界更平滑。
    """

    def __init__(self, rate: float, capacity: int):
        """
        Args:
            rate:     每秒产生的令牌数
            capacity: 桶的最大容量（突发上限）
        """
        self._rate = rate
        self._capacity = capacity
        self._tokens = float(capacity)
        self._last_refill = time.monotonic()

    def consume(self, tokens: int = 1) -> bool:
        """尝试消费令牌"""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(
            self._capacity,
            self._tokens + elapsed * self._rate,
        )
        self._last_refill = now

        if self._tokens >= tokens:
            self._tokens -= tokens
            return True
        return False


class RateLimiter:
    """
    多级限流器。

    四级限流策略：
    1. 全局限流：保护整个系统不被压垮
    2. 用户级限流：防止单个用户滥用
    3. 会话级限流：控制单会话消息频率
    4. 端点级限流：对特定 API 做精细控制

    Usage:
        limiter = RateLimiter()
        limiter.add_rule("global", max_requests=1000, window_seconds=60)
        limiter.add_rule("user", max_requests=60, window_seconds=60)
        limiter.add_rule("session", max_requests=20, window_seconds=60)
        limiter.add_rule("endpoint:/chat/completions", max_requests=10, window_seconds=60)

        allowed, meta = limiter.check(
            level="user",
            identifier=user_id,
        )
    """

    def __init__(self):
        self._rules: dict[str, RateLimitRule] = {}
        # (level, identifier) -> TokenBucket
        self._buckets: dict[tuple[str, str], TokenBucket] = defaultdict(
            lambda: None  # lazy init
        )

    def add_rule(self, name: str, max_requests: int, window_seconds: float):
        """添加限流规则"""
        rate = max_requests / window_seconds
        self._rules[name] = RateLimitRule(
            max_requests=max_requests,
            window_seconds=window_seconds,
        )

    def check(self, level: str, identifier: str) -> tuple[bool, dict]:
        """
        检查请求是否允许通过。

        Args:
            level:       限流级别 ("global", "user", "session", "endpoint:*")
            identifier:  标识符（用户ID、会话ID、端点路径等）

        Returns:
            (allowed, metadata) - metadata 包含剩余请求数、重置时间等
        """
        rule = self._rules.get(level)
        if rule is None:
            return True, {}

        key = (level, identifier)
        now = time.monotonic()

        # Lazy init token bucket
        if self._buckets[key] is None:
            rate = rule.max_requests / rule.window_seconds
            self._buckets[key] = TokenBucket(rate, rule.max_requests)

        allowed = self._buckets[key].consume()

        remaining = int(self._buckets[key]._tokens)
        reset_at = now + rule.window_seconds

        return allowed, {
            "limit": rule.max_requests,
            "remaining": max(0, remaining),
            "reset_at": reset_at,
            "window_seconds": rule.window_seconds,
        }


# ---------- FastAPI Middleware ----------

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse
from starlette.status import HTTP_429_TOO_MANY_REQUESTS


class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI 限流中间件"""

    def __init__(self, app, limiter: RateLimiter):
        super().__init__(app)
        self._limiter = limiter

    async def dispatch(self, request: Request, call_next) -> Response:
        # 1. 全局限流
        allowed, _ = self._limiter.check("global", "system")
        if not allowed:
            return JSONResponse(
                status_code=HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": "Server overloaded. Please try again later."},
                headers={"Retry-After": "60"},
            )

        # 2. 用户级限流（从 JWT 提取 user_id）
        user_id = request.state.user_id if hasattr(request.state, 'user_id') else "anonymous"
        allowed, meta = self._limiter.check("user", user_id)
        if not allowed:
            return JSONResponse(
                status_code=HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": "Rate limit exceeded. Try again later."},
                headers={
                    "X-RateLimit-Limit": str(meta["limit"]),
                    "X-RateLimit-Remaining": str(meta["remaining"]),
                    "Retry-After": str(int(meta["window_seconds"])),
                },
            )

        # 3. 端点级限流
        endpoint_key = f"endpoint:{request.url.path}"
        allowed, meta = self._limiter.check(endpoint_key, user_id)
        if not allowed:
            return JSONResponse(
                status_code=HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": f"Rate limit exceeded for {request.url.path}"},
                headers={
                    "X-RateLimit-Limit": str(meta["limit"]),
                    "X-RateLimit-Remaining": str(meta["remaining"]),
                    "Retry-After": str(int(meta["window_seconds"])),
                },
            )

        response = await call_next(request)
        # 在响应头中注入限流信息
        if meta:
            response.headers["X-RateLimit-Limit"] = str(meta["limit"])
            response.headers["X-RateLimit-Remaining"] = str(meta["remaining"])
        return response
```

---

## 5. 输入校验 `security/validation.py`

```python
from typing import Any
import re


class InputValidator:
    """
    输入安全校验器。

    防护目标：
    - Prompt Injection：检测并标记可疑的注入尝试
    - SQL/NoSQL Injection：过滤特殊字符模式
    - XSS：转义 HTML 标签（前端渲染场景）
    - 长度限制：防止超大输入导致资源耗尽
    """

    # 常见 prompt injection 模式
    INJECTION_PATTERNS = [
        r"(?i)\bignore\s+previous\s+instructions\b",
        r"(?i)\bsystem\s*:\s*",
        r"(?i)\byou\s+are\s+now\s+",
        r"(?i)\bdisregard\s+(the\s+)?previous\b",
        r"(?i)\<\/?xml\b",           # XML injection
        r"(?i)\bDROP\s+TABLE\b",     # SQL injection
    ]

    @classmethod
    def validate_message(
        cls,
        text: str,
        max_length: int = 10000,
        check_injection: bool = True,
    ) -> tuple[bool, list[str]]:
        """
        校验用户消息。

        Returns:
            (is_valid, warnings) - warnings 包含检测到的可疑模式
        """
        warnings: list[str] = []

        # 长度检查
        if len(text) > max_length:
            return False, [f"Message exceeds maximum length of {max_length} characters"]

        # Prompt injection 检测（仅警告，不阻断）
        if check_injection:
            for pattern in cls.INJECTION_PATTERNS:
                if re.search(pattern, text):
                    warnings.append(f"Suspicious pattern detected: {pattern}")
                    break  # 只报告一次

        return True, warnings

    @classmethod
    def sanitize_html(cls, text: str) -> str:
        """转义 HTML 特殊字符"""
        import html
        return html.escape(text, quote=True)

    @classmethod
    def validate_filename(cls, filename: str) -> tuple[bool, str]:
        """校验文件名安全性"""
        if not filename or len(filename) > 255:
            return False, "Invalid filename"

        # 防止路径遍历
        if ".." in filename or "/" in filename or "\\" in filename:
            return False, "Filename contains invalid characters"

        return True, ""


# ---------- Prompt Injection 防护层 ----------

class PromptGuard:
    """
    System prompt 注入防护。

    在 Agent Core 组装 messages 时，对用户输入进行隔离处理：
    - 用户消息永远放在 user role 中，不会混入 system prompt
    - 对工具返回结果做二次校验，防止工具输出被注入
    """

    @staticmethod
    def isolate_user_input(user_message: str) -> dict:
        """将用户输入隔离为标准的 message 格式"""
        return {
            "role": "user",
            "content": user_message,
        }

    @staticmethod
    def sanitize_tool_output(tool_result: Any) -> str:
        """清理工具输出，防止注入到后续对话"""
        if isinstance(tool_result, dict):
            import json
            text = json.dumps(tool_result, ensure_ascii=False)
        else:
            text = str(tool_result)

        # 移除可能的控制字符
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        return text
```

---

## 6. 审计日志 `security/audit.py`

```python
import time
import logging
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class AuditEntry:
    """审计日志条目"""
    timestamp: float
    user_id: str
    action: str                    # 操作类型：login, create_session, delete_memory, etc.
    resource_type: str             # 资源类型：session, message, tool, plugin, etc.
    resource_id: Optional[str]     # 资源 ID
    success: bool                  # 是否成功
    ip_address: Optional[str]      # 客户端 IP
    user_agent: Optional[str]      # User-Agent
    details: dict | None = None    # 附加信息


class AuditLogger:
    """
    审计日志记录器。

    记录所有关键操作，用于：
    - 安全事件追溯
    - 合规审查（GDPR、等保等）
    - 异常行为检测

    存储策略：
    - 开发环境：内存 + 文件日志
    - 生产环境：数据库表 + 异步写入队列
    """

    def __init__(self, log_file: str | None = None):
        self._logger = logging.getLogger("audit")
        if log_file:
            handler = logging.FileHandler(log_file)
            handler.setFormatter(logging.JSONFormatter())
            self._logger.addHandler(handler)
        self._logger.setLevel(logging.INFO)

    def log(
        self,
        user_id: str,
        action: str,
        resource_type: str,
        resource_id: Optional[str] = None,
        success: bool = True,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        details: dict | None = None,
    ):
        """记录审计日志"""
        entry = AuditEntry(
            timestamp=time.time(),
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            success=success,
            ip_address=ip_address,
            user_agent=user_agent,
            details=details,
        )
        self._logger.info(
            "",  # msg 为空，所有信息在 extra 中
            extra={k: str(v) for k, v in entry.__dict__.items()},
        )

    def log_api_call(
        self,
        user_id: str,
        method: str,
        path: str,
        status_code: int,
        ip_address: Optional[str] = None,
        duration_ms: float | None = None,
    ):
        """记录 API 调用审计"""
        self.log(
            user_id=user_id,
            action=f"{method} {path}",
            resource_type="api",
            resource_id=path,
            success=200 <= status_code < 400,
            ip_address=ip_address,
            details={"status_code": status_code, "duration_ms": duration_ms},
        )


# ---------- FastAPI Middleware ----------

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class AuditMiddleware(BaseHTTPMiddleware):
    """FastAPI 审计中间件"""

    def __init__(self, app, audit_logger: AuditLogger):
        super().__init__(app)
        self._audit = audit_logger

    async def dispatch(self, request: Request, call_next) -> Response:
        start_time = time.time()

        user_id = getattr(request.state, 'user_id', 'anonymous')
        ip_address = request.client.host if request.client else None
        user_agent = request.headers.get("user-agent")

        response = await call_next(request)

        duration_ms = (time.time() - start_time) * 1000

        self._audit.log_api_call(
            user_id=user_id,
            method=request.method,
            path=str(request.url.path),
            status_code=response.status_code,
            ip_address=ip_address,
            duration_ms=duration_ms,
        )

        return response
```

---

## 7. 安全配置 `config/security.yaml`

```yaml
# config/settings.yaml (新增 security section)
security:
  jwt:
    algorithm: "HS256"
    access_token_minutes: 15
    refresh_token_days: 7
    secret_key_env: "JWT_SECRET_KEY"   # 从环境变量读取密钥

  rate_limit:
    global:
      max_requests: 1000
      window_seconds: 60
    user:
      max_requests: 60
      window_seconds: 60
    session:
      max_requests: 20
      window_seconds: 60
    endpoints:
      "/api/chat/completions":
        max_requests: 10
        window_seconds: 60
      "/api/tools/execute":
        max_requests: 30
        window_seconds: 60

  input:
    max_message_length: 10000
    check_prompt_injection: true
    sanitize_html: false              # 默认不转义（Agent 不需要 HTML）

  audit:
    enabled: true
    log_file: "logs/audit.log"        # 生产环境改为数据库存储
```

---

## 8. 架构总览

```
                    ┌─────────────────────┐
                    │   HTTP Request      │
                    └──────────┬──────────┘
                               ▼
              ┌────────────────────────────────┐
              │     RateLimitMiddleware        │ ← 四级限流检查
              │  global → user → session → ep  │
              └────────────┬───────────────────┘
                           ▼
              ┌────────────────────────────────┐
              │      AuthMiddleware            │ ← JWT 验证 + RBAC
              │  parse token → inject role     │
              └────────────┬───────────────────┘
                           ▼
              ┌────────────────────────────────┐
              │    InputValidator              │ ← 输入校验 + injection 检测
              └────────────┬───────────────────┘
                           ▼
                    ┌─────────────┐
                    │   Router    │ → FastAPI endpoints
                    └──────┬──────┘
                           ▼
              ┌────────────────────────────────┐
              │     AuditMiddleware            │ ← 审计日志记录
              └────────────────────────────────┘

         ┌─────────────────────────────────────┐
         │        Security Components          │
         ├─────────────────────────────────────┤
         │ JWTManager    → Token lifecycle     │
         │ PermissionMatrix → RBAC checks      │
         │ RateLimiter   → Token bucket algo   │
         │ InputValidator → Sanitization       │
         │ PromptGuard   → Injection defense   │
         │ AuditLogger   → Compliance logging  │
         └─────────────────────────────────────┘
```

---

## 9. 设计总结

| 特性 | 实现方式 |
|------|---------|
| **RBAC** | 三级角色（public/authenticated/admin）+ 资源操作矩阵 |
| **JWT 认证** | access_token (15min) + refresh_token (7d)，支持撤销 |
| **多级限流** | Token Bucket 算法，全局/用户/会话/端点四级控制 |
| **输入校验** | Prompt injection 检测、长度限制、文件名安全校验 |
| **审计日志** | 结构化 JSON 格式，中间件自动记录 API 调用轨迹 |
| **FastAPI 集成** | Middleware + Depends 依赖注入，声明式权限控制 |

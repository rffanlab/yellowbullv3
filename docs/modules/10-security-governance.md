# 安全与治理详细设计（Security & Governance）

## 1. 职责边界

| 领域 | 说明 |
|------|------|
| **认证授权** | API Key、JWT Token、RBAC 角色权限控制 |
| **内容过滤** | 输入/输出双向安全检查，敏感词 + AI 审核 |
| **Prompt 注入防护** | 检测并阻断恶意 prompt 攻击 |
| **审计日志** | 所有操作的不可篡改记录 |
| **速率限制** | API 调用频率控制，防滥用 |

---

## 2. RBAC 权限系统 `security/rbac.py`

```python
"""
基于角色的访问控制（RBAC）。

角色层级：
    admin → manager → user → viewer

权限矩阵：
| 操作              | admin | manager | user | viewer |
|-------------------|-------|---------|------|--------|
| create_workspace  | ✓     | -       | -    | -      |
| manage_users      | ✓     | ✓       | -    | -      |
| create_agent      | ✓     | ✓       | ✓    | -      |
| run_agent         | ✓     | ✓       | ✓    | ✓      |
| view_audit_log    | ✓     | ✓       | -    | -      |
| manage_billing    | ✓     | -       | -    | -      |
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class Role(str, Enum):
    ADMIN = "admin"
    MANAGER = "manager"
    USER = "user"
    VIEWER = "viewer"


class Permission(str, Enum):
    CREATE_WORKSPACE = "workspace:create"
    DELETE_WORKSPACE = "workspace:delete"
    MANAGE_USERS = "users:manage"
    CREATE_AGENT = "agent:create"
    UPDATE_AGENT = "agent:update"
    DELETE_AGENT = "agent:delete"
    RUN_AGENT = "agent:run"
    VIEW_AUDIT_LOG = "audit:view"
    MANAGE_BILLING = "billing:manage"
    VIEW_BILLING = "billing:view"
    MANAGE_TOOLS = "tools:manage"
    USE_TOOL = "tools:use"


# 角色 → 权限映射
ROLE_PERMISSIONS: dict[Role, set[Permission]] = {
    Role.ADMIN: {p for p in Permission},
    Role.MANAGER: {
        Permission.CREATE_AGENT, Permission.UPDATE_AGENT, Permission.DELETE_AGENT,
        Permission.RUN_AGENT, Permission.MANAGE_USERS, Permission.VIEW_AUDIT_LOG,
        Permission.VIEW_BILLING, Permission.USE_TOOL,
    },
    Role.USER: {
        Permission.CREATE_AGENT, Permission.UPDATE_AGENT,
        Permission.RUN_AGENT, Permission.USE_TOOL, Permission.VIEW_BILLING,
    },
    Role.VIEWER: {
        Permission.RUN_AGENT,
    },
}


@dataclass
class User:
    user_id: str
    username: str
    role: Role = Role.USER
    workspace_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResourceContext:
    """资源上下文（用于权限检查）"""
    resource_type: str       # "agent" | "workspace" | "tool" | ...
    resource_id: str         # 资源 ID
    workspace_id: str | None = None  # 所属工作空间


class RBACGuard:
    """RBAC 权限守卫"""

    def __init__(self):
        self._users: dict[str, User] = {}

    def register_user(self, user: User):
        self._users[user.user_id] = user

    def check_permission(
        self, user_id: str, permission: Permission, context: ResourceContext | None = None
    ) -> bool:
        """检查用户是否有某项权限"""
        user = self._users.get(user_id)
        if not user:
            return False

        user_permissions = ROLE_PERMISSIONS.get(user.role, set())
        if permission not in user_permissions:
            logger.warning(
                f"Permission denied: user={user.username} role={user.role} "
                f"requested={permission.value}"
            )
            return False

        # 工作空间隔离检查
        if context and context.workspace_id:
            if context.workspace_id not in user.workspace_ids:
                logger.warning(
                    f"Workspace access denied: user={user.username} "
                    f"workspace={context.workspace_id}"
                )
                return False

        return True

    def require_permission(
        self, user_id: str, permission: Permission, context: ResourceContext | None = None
    ):
        """权限检查失败时抛出异常"""
        if not self.check_permission(user_id, permission, context):
            raise PermissionError(
                f"User '{user_id}' lacks permission '{permission.value}'"
            )

    def get_user_permissions(self, user_id: str) -> set[Permission]:
        """获取用户的所有权限"""
        user = self._users.get(user_id)
        return ROLE_PERMISSIONS.get(user.role, set()) if user else set()


class WorkspaceGuard:
    """工作空间隔离守卫"""

    def __init__(self):
        # workspace_id → set of user_ids
        self._memberships: dict[str, set[str]] = {}

    def add_member(self, workspace_id: str, user_id: str):
        if workspace_id not in self._memberships:
            self._memberships[workspace_id] = set()
        self._memberships[workspace_id].add(user_id)

    def can_access(self, user_id: str, workspace_id: str) -> bool:
        return user_id in self._memberships.get(workspace_id, set())

    def list_workspaces(self, user_id: str) -> list[str]:
        """列出用户有权限访问的工作空间"""
        return [
            ws_id for ws_id, members in self._memberships.items()
            if user_id in members
        ]
```

---

## 3. API Key 认证 `security/auth.py`

```python
"""
API Key 认证中间件。

支持：
- Header: X-API-Key
- Query param: ?api_key=...
- Bearer Token (JWT)
"""

import hashlib
import hmac
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class APIKeyRecord:
    """API Key 记录（存储时只存哈希）"""
    key_id: str
    key_hash: str           # SHA-256 hash of the actual key
    user_id: str
    workspace_id: str | None
    permissions: list[str]  # 权限列表
    rate_limit_rpm: int     # 每分钟请求数限制
    created_at: float
    expires_at: float | None  # 过期时间（None = 永不过期）
    is_active: bool = True


class APIKeyStore:
    """API Key 存储"""

    def __init__(self):
        self._keys: dict[str, APIKeyRecord] = {}  # key_id → record

    def add_key(self, record: APIKeyRecord):
        self._keys[record.key_id] = record

    def verify(self, api_key: str) -> APIKeyRecord | None:
        """验证 API Key（通过哈希比对）"""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        for record in self._keys.values():
            if hmac.compare_digest(record.key_hash, key_hash):
                return record
        return None


class JWTManager:
    """JWT Token 管理"""

    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self._secret = secret_key
        self._algorithm = algorithm

    def create_token(self, user_id: str, permissions: list[str], expires_in: int = 3600) -> str:
        """创建 JWT Token"""
        import jwt
        payload = {
            "sub": user_id,
            "perms": permissions,
            "iat": time.time(),
            "exp": time.time() + expires_in,
        }
        return jwt.encode(payload, self._secret, algorithm=self._algorithm)

    def verify_token(self, token: str) -> dict[str, Any]:
        """验证并解析 JWT Token"""
        import jwt
        try:
            payload = jwt.decode(token, self._secret, algorithms=[self._algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token expired")
            raise PermissionError("Token expired")
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid JWT token: {e}")
            raise PermissionError("Invalid token")


class AuthMiddleware:
    """
    认证中间件（FastAPI / Starlette）。

    Usage:
        auth = AuthMiddleware(api_key_store=store, jwt_manager=jwt_mgr)
        # In FastAPI dependency:
        # user_id = await auth.authenticate(request)
    """

    def __init__(self, api_key_store: APIKeyStore, jwt_manager: JWTManager | None = None):
        self._api_keys = api_key_store
        self._jwt = jwt_manager

    async def authenticate(self, request: Any) -> str:
        """
        从请求中提取并验证认证信息，返回 user_id。

        优先级：JWT Bearer > X-API-Key header > api_key query param
        """
        # 1. JWT Bearer Token
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer ") and self._jwt:
            token = auth_header[7:]
            payload = self._jwt.verify_token(token)
            return payload["sub"]

        # 2. X-API-Key header
        api_key = request.headers.get("X-API-Key") or request.query_params.get("api_key")
        if api_key:
            record = self._api_keys.verify(api_key)
            if not record:
                raise PermissionError("Invalid API key")
            if not record.is_active:
                raise PermissionError("API key is deactivated")
            if record.expires_at and time.time() > record.expires_at:
                raise PermissionError("API key has expired")
            return record.user_id

        raise PermissionError("No authentication provided")
```

---

## 4. 速率限制 `security/rate_limit.py`

```python
"""
速率限制器。

支持：
- 滑动窗口计数器（精确）
- Token Bucket（平滑限流）
"""

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """速率限制配置"""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    burst_size: int = 10          # Token Bucket 突发容量


class SlidingWindowLimiter:
    """滑动窗口计数器限流器"""

    def __init__(self):
        # key → list of timestamps
        self._requests: dict[str, list[float]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def is_allowed(self, key: str, config: RateLimitConfig) -> tuple[bool, dict]:
        """
        检查请求是否允许。

        Returns: (allowed, info_dict)
        """
        now = time.time()
        async with self._lock:
            timestamps = self._requests[key]

            # 清理过期记录（1小时窗口）
            hour_ago = now - 3600
            timestamps[:] = [t for t in timestamps if t > hour_ago]

            # 检查分钟级限制
            minute_ago = now - 60
            minute_count = sum(1 for t in timestamps if t > minute_ago)

            # 检查小时级限制
            hour_count = len(timestamps)

            info = {
                "minute_count": minute_count,
                "minute_limit": config.requests_per_minute,
                "hour_count": hour_count,
                "hour_limit": config.requests_per_hour,
                "retry_after": 0,
            }

            if minute_count >= config.requests_per_minute:
                # 计算需要等待的时间
                oldest_in_minute = next((t for t in sorted(timestamps) if t > minute_ago), now)
                info["retry_after"] = max(1, int(60 - (now - oldest_in_minute)))
                logger.warning(f"Rate limit exceeded (minute): key={key}")
                return False, info

            if hour_count >= config.requests_per_hour:
                info["retry_after"] = 3600
                logger.warning(f"Rate limit exceeded (hour): key={key}")
                return False, info

            # 记录本次请求
            timestamps.append(now)
            return True, info


class TokenBucketLimiter:
    """Token Bucket 限流器（适合平滑限流）"""

    def __init__(self):
        # key → {tokens, last_refill}
        self._buckets: dict[str, dict] = {}
        self._lock = asyncio.Lock()

    async def is_allowed(
        self, key: str, rate: float = 10.0, burst: int = 20
    ) -> tuple[bool, dict]:
        """
        检查并消费一个 token。

        Args:
            rate: 每秒补充的 token 数
            burst: bucket 最大容量
        """
        now = time.time()
        async with self._lock:
            if key not in self._buckets:
                self._buckets[key] = {"tokens": burst, "last_refill": now}

            bucket = self._buckets[key]
            # 补充 token
            elapsed = now - bucket["last_refill"]
            bucket["tokens"] = min(burst, bucket["tokens"] + elapsed * rate)
            bucket["last_refill"] = now

            if bucket["tokens"] >= 1:
                bucket["tokens"] -= 1
                return True, {"remaining_tokens": bucket["tokens"]}
            else:
                retry_after = (1 - bucket["tokens"]) / rate
                return False, {"retry_after": retry_after}


class RateLimitMiddleware:
    """速率限制中间件"""

    def __init__(self):
        self._sliding_window = SlidingWindowLimiter()
        self._token_bucket = TokenBucketLimiter()
        self._configs: dict[str, RateLimitConfig] = {}  # user_id → config

    def set_config(self, user_id: str, config: RateLimitConfig):
        self._configs[user_id] = config

    async def check_rate_limit(
        self, user_id: str, endpoint: str = "/"
    ) -> tuple[bool, dict]:
        """检查速率限制"""
        config = self._configs.get(user_id, RateLimitConfig())
        key = f"{user_id}:{endpoint}"

        # 先用 Token Bucket 做快速判断（突发流量）
        allowed, info = await self._token_bucket.is_allowed(
            key, rate=config.requests_per_minute / 60, burst=config.burst_size
        )

        if not allowed:
            return False, info

        # 再用滑动窗口做精确统计
        allowed, info = await self._sliding_window.is_allowed(key, config)
        return allowed, info
```

---

## 5. 内容安全过滤 `security/content_filter.py`

```python
"""
内容安全过滤器。

双向检查：
- Input Filter: 用户输入 → 检测敏感词、prompt 注入、恶意代码
- Output Filter: Agent 输出 → 检测不当内容、信息泄露
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ThreatLevel(str, Enum):
    SAFE = "safe"
    LOW = "low"           # 可疑，可警告通过
    MEDIUM = "medium"     # 需要人工审核
    HIGH = "high"         # 直接阻断


@dataclass
class FilterResult:
    """过滤结果"""
    is_safe: bool
    threat_level: ThreatLevel
    blocked_reasons: list[str] = field(default_factory=list)
    sanitized_content: str = ""   # 清洗后的内容（可选）
    metadata: dict[str, Any] = field(default_factory=dict)


class KeywordFilter:
    """关键词过滤器"""

    def __init__(self):
        self._blocked_patterns: list[re.Pattern] = []
        self._sensitive_words: set[str] = set()

    def add_blocked_pattern(self, pattern: str, flags: int = re.IGNORECASE):
        self._blocked_patterns.append(re.compile(pattern, flags))

    def add_sensitive_words(self, words: list[str]):
        self._sensitive_words.update(w.lower() for w in words)

    def check(self, content: str) -> FilterResult:
        """检查内容是否包含敏感词或匹配阻断模式"""
        reasons = []
        content_lower = content.lower()

        # 精确关键词匹配
        found_words = self._sensitive_words.intersection(
            word.lower() for word in re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+', content)
        )
        if found_words:
            reasons.append(f"Contains sensitive words: {found_words}")

        # 正则模式匹配
        for pattern in self._blocked_patterns:
            if pattern.search(content):
                reasons.append(f"Matches blocked pattern: {pattern.pattern}")

        if reasons:
            return FilterResult(
                is_safe=False,
                threat_level=ThreatLevel.HIGH,
                blocked_reasons=reasons,
            )

        return FilterResult(is_safe=True, threat_level=ThreatLevel.SAFE)


class PromptInjectionDetector:
    """Prompt 注入攻击检测器"""

    # 常见 prompt 注入模式
    INJECTION_PATTERNS = [
        r"(?i)\bignore\s+previous\s+instructions\b",
        r"(?i)\bdisregard\s+the\s+above\b",
        r"(?i)system\s*:\s*",
        r"(?i)<\s*system\s*>",
        r"(?i)\byour\s+task\s+is\s+now\b",
        r"(?i)\bforg(?:et|ive)\s+(?:all\s+)?instructions?\b",
        r"(?i)repeat\s+the\s+above\s*(?:text|prompt|instructions)",
        r"(?i)output\s+(?:your\s+)?system\s*(?:prompt|message|instructions)",
        r"(?i)\`{3,}\s*system",           # ```system 代码块注入
        r"(?i)^[\s\n]*(?:ACTIVATE|ENABLE|SET)\s+(?:DEVELOPER|DEBUG|ADMIN)\s+MODE",
    ]

    def __init__(self):
        self._patterns = [re.compile(p) for p in self.INJECTION_PATTERNS]

    def check(self, content: str) -> FilterResult:
        """检测 prompt 注入攻击"""
        reasons = []

        for pattern in self._patterns:
            match = pattern.search(content)
            if match:
                reasons.append(f"Possible prompt injection: '{match.group()[:50]}'")

        # 检查嵌套指令（多层引号包裹的指令）
        nested_count = content.count('"') + content.count("'")
        if nested_count > 10 and len(content) < 2000:
            reasons.append("Excessive quoting detected (possible injection)")

        if not reasons:
            return FilterResult(is_safe=True, threat_level=ThreatLevel.SAFE)

        # 多条命中 → HIGH，单条命中 → MEDIUM
        level = ThreatLevel.HIGH if len(reasons) >= 2 else ThreatLevel.MEDIUM
        return FilterResult(
            is_safe=False,
            threat_level=level,
            blocked_reasons=reasons,
        )


class OutputContentFilter:
    """输出内容过滤器"""

    # 检测可能的信息泄露模式
    LEAK_PATTERNS = [
        r"(?i)\bapi[_-]?key\s*[:=]\s*\S+",       # API Key 泄露
        r"(?i)\bpassword\s*[:=]\s*\S+",           # 密码泄露
        r"\b[A-Za-z0-9+/]{40,}={0,2}\b",          # Base64 encoded secrets
        r"sk-[A-Za-z0-9]{20,}",                   # OpenAI API key pattern
    ]

    def __init__(self):
        self._patterns = [re.compile(p) for p in self.LEAK_PATTERNS]

    def check(self, content: str) -> FilterResult:
        """检查输出内容是否包含敏感信息"""
        reasons = []

        for pattern in self._patterns:
            match = pattern.search(content)
            if match:
                # 脱敏显示
                matched = match.group()[:20] + "..."
                reasons.append(f"Possible info leak detected near: '{matched}'")

        if not reasons:
            return FilterResult(is_safe=True, threat_level=ThreatLevel.SAFE)

        level = ThreatLevel.MEDIUM
        sanitized = content
        for pattern in self._patterns:
            sanitized = pattern.sub("[REDACTED]", sanitized)

        return FilterResult(
            is_safe=False,
            threat_level=level,
            blocked_reasons=reasons,
            sanitized_content=sanitized,
        )


class ContentSecurityFilter:
    """
    综合内容安全过滤器。

    Usage:
        filter = ContentSecurityFilter()
        # Check user input
        result = await filter.check_input(user_message)
        if not result.is_safe:
            raise SecurityError(result.blocked_reasons)

        # Check agent output
        result = await filter.check_output(agent_response)
    """

    def __init__(self):
        self._keyword_filter = KeywordFilter()
        self._injection_detector = PromptInjectionDetector()
        self._output_filter = OutputContentFilter()

        # 默认敏感词
        self._keyword_filter.add_sensitive_words([
            "暴力", "恐怖", "自杀", "毒品", "赌博",
        ])
        self._keyword_filter.add_blocked_pattern(r"(?i)\b(?:eval|exec)\s*\(")

    async def check_input(self, content: str) -> FilterResult:
        """检查用户输入"""
        results = [
            self._keyword_filter.check(content),
            self._injection_detector.check(content),
        ]

        for result in results:
            if not result.is_safe and result.threat_level == ThreatLevel.HIGH:
                return result

        # 合并所有警告
        all_reasons = []
        max_level = ThreatLevel.SAFE
        for result in results:
            all_reasons.extend(result.blocked_reasons)
            if result.threat_level.value > max_level.value:
                max_level = result.threat_level

        return FilterResult(
            is_safe=len(all_reasons) == 0,
            threat_level=max_level,
            blocked_reasons=all_reasons,
        )

    async def check_output(self, content: str) -> FilterResult:
        """检查 Agent 输出"""
        result = self._output_filter.check(content)
        return result
```

---

## 6. 审计日志 `security/audit.py`

```python
"""
审计日志系统。

记录所有关键操作，支持查询和导出。
"""

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)


@dataclass
class AuditEntry:
    """审计日志条目"""
    timestamp: float
    user_id: str
    action: str             # "api_call" | "agent_run" | "config_change" | ...
    resource_type: str      # "agent" | "workspace" | "tool" | ...
    resource_id: str        # 资源 ID
    outcome: str            # "success" | "failure" | "denied"
    details: dict[str, Any] = field(default_factory=dict)
    ip_address: str = ""
    user_agent: str = ""


class BaseAuditStore(Protocol):
    """审计存储后端协议"""

    async def append(self, entry: AuditEntry) -> bool: ...
    async def query(
        self,
        user_id: str | None = None,
        action: str | None = None,
        resource_type: str | None = None,
        since: float | None = None,
        until: float | None = None,
        limit: int = 100,
    ) -> list[AuditEntry]: ...


class JSONAuditStore(BaseAuditStore):
    """JSON 文件审计存储（追加写入）"""

    def __init__(self, log_dir: str = "./data/audit"):
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)

    async def append(self, entry: AuditEntry) -> bool:
        """按日期分文件追加写入"""
        date_str = time.strftime("%Y-%m-%d", time.localtime(entry.timestamp))
        log_file = self._log_dir / f"audit_{date_str}.jsonl"

        line = json.dumps(asdict(entry), ensure_ascii=False) + "\n"
        log_file.write_text(line, mode="a", encoding="utf-8")
        return True

    async def query(
        self, user_id=None, action=None, resource_type=None,
        since=None, until=None, limit=100
    ) -> list[AuditEntry]:
        """查询审计日志"""
        entries = []
        for log_file in sorted(self._log_dir.glob("audit_*.jsonl")):
            for line in log_file.read_text(encoding="utf-8").strip().split("\n"):
                if not line:
                    continue
                data = json.loads(line)
                entry = AuditEntry(**data)

                # 过滤条件
                if user_id and entry.user_id != user_id:
                    continue
                if action and entry.action != action:
                    continue
                if resource_type and entry.resource_type != resource_type:
                    continue
                if since and entry.timestamp < since:
                    continue
                if until and entry.timestamp > until:
                    continue

                entries.append(entry)
                if len(entries) >= limit:
                    return entries

        return entries


class AuditLogger:
    """
    审计日志记录器。

    Usage:
        audit = AuditLogger(store=JSONAuditStore())
        await audit.log_action(
            user_id="user_001",
            action="agent_run",
            resource_type="agent",
            resource_id="agent_abc",
            outcome="success",
            details={"model": "gpt-4o", "tokens_used": 500},
        )
    """

    def __init__(self, store: BaseAuditStore):
        self._store = store

    async def log_action(
        self,
        user_id: str,
        action: str,
        resource_type: str,
        resource_id: str,
        outcome: str,
        details: dict[str, Any] | None = None,
        ip_address: str = "",
        user_agent: str = "",
    ):
        entry = AuditEntry(
            timestamp=time.time(),
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            outcome=outcome,
            details=details or {},
            ip_address=ip_address,
            user_agent=user_agent,
        )
        await self._store.append(entry)

    async def log_api_call(
        self, user_id: str, method: str, path: str, status_code: int,
        duration_ms: float, details: dict[str, Any] | None = None,
    ):
        """记录 API 调用"""
        await self.log_action(
            user_id=user_id,
            action="api_call",
            resource_type="api",
            resource_id=f"{method} {path}",
            outcome="success" if status_code < 400 else "failure",
            details={**(details or {}), "method": method, "status": status_code, "duration_ms": duration_ms},
        )

    async def log_security_event(
        self, user_id: str, event_type: str, severity: str,
        details: dict[str, Any] | None = None,
    ):
        """记录安全事件"""
        await self.log_action(
            user_id=user_id,
            action=f"security:{event_type}",
            resource_type="security",
            resource_id=event_type,
            outcome=severity,
            details=details or {},
        )
```

---

## 7. YAML 配置 `config/security.yaml`

```yaml
security:
  auth:
    api_key:
      enabled: true
      header_name: "X-API-Key"

    jwt:
      enabled: true
      secret_key: "${JWT_SECRET}"       # 环境变量引用
      algorithm: "HS256"
      token_expiry_seconds: 3600        # 1 hour

  rbac:
    default_role: "user"                # 新用户默认角色
    workspace_isolation: true           # 工作空间隔离

  rate_limit:
    enabled: true
    default:
      requests_per_minute: 60
      requests_per_hour: 1000
      burst_size: 10
    tiers:
      free: { rpm: 20, rph: 500 }
      pro: { rpm: 60, rph: 3000 }
      enterprise: { rpm: 200, rph: 10000 }

  content_filter:
    input_filter:
      enabled: true
      sensitive_words_file: "./config/sensitive_words.txt"
      block_code_execution: true        # 阻止 eval/exec 等危险操作

    prompt_injection:
      enabled: true
      threat_level_action:              # 不同威胁级别的处理方式
        medium: "warn_and_pass"         # warn_and_pass | block
        high: "block"

    output_filter:
      enabled: true
      redact_secrets: true              # 自动脱敏 API Key、密码等

  audit:
    enabled: true
    store_type: "json_file"             # json_file | database
    log_dir: "./data/audit"
    retention_days: 90                  # 日志保留天数
```

---

## 8. 架构总览

```
                    ┌─────────────────────┐
                    │     HTTP Request     │
                    └──────────┬──────────┘
                               ▼
              ┌──────────────────────────────────┐
              │      Auth Middleware             │
              │  JWT / API Key → user_id         │
              └──────────┬───────────────────────┘
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼               ▼
   ┌────────────┐ ┌────────────┐ ┌────────────┐
   │ Rate Limit │ │ RBAC Check │ │ Content    │
   │ (Token     │ │ (role +    │ │ Filter     │
   │  Bucket)   │ │  workspace)│ │ (input)    │
   └────────────┘ └────────────┘ └────────────┘
                         │
                         ▼ All passed
              ┌───────────────────────────┐
              │       Agent Core          │
              │                           │
              │  Content Filter (output)  │
              └──────────┬────────────────┘
                         │
                         ▼
              ┌───────────────────────────┐
              │      Audit Logger         │
              │  (append-only log)        │
              └───────────────────────────┘
```

---

## 9. 设计总结

| 特性 | 实现方式 |
|------|---------|
| **认证** | API Key（哈希存储）+ JWT Token，中间件统一处理 |
| **授权** | RBAC 四级角色 + 工作空间隔离 |
| **速率限制** | Token Bucket（突发流量）+ Sliding Window（精确统计） |
| **内容过滤** | 关键词 + Prompt 注入检测 + 输出脱敏 |
| **审计日志** | JSONL 追加写入，按日期分文件，支持多条件查询 |

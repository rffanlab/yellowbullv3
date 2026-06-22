# 安全与治理（Security & Governance）详细设计

## 1. 职责边界

| 支柱 | 说明 |
|------|------|
| **API Key 认证** | X-API-Key header 验证，支持多 key 管理 |
| **RBAC 权限控制** | 用户/角色/资源三级权限模型 |
| **内容过滤** | 输入输出敏感内容检测与拦截 |
| **Prompt 注入防护** | 系统提示隔离、输入清洗 |
| **速率限制** | 按 API key / IP 限流 |
| **审计日志** | 操作记录 + 合规报告 |

---

## 2. API Key 认证 `security/auth.py`

```python
"""
API Key 认证中间件。

支持：
- X-API-Key header 验证
- 多 key 管理（不同 key 对应不同权限级别）
- Key 轮换与过期控制
"""

import hashlib
import hmac
from datetime import datetime, timezone, timedelta
from typing import Optional


class APIKey:
    """API Key 实体"""

    def __init__(
        self, name: str, secret: str, roles: list[str] = None,
        expires_at: datetime = None, rate_limit_rpm: int = 60,
    ):
        self.name = name
        self.secret_hash = hashlib.sha256(secret.encode()).hexdigest()
        self.roles = roles or ["user"]
        self.expires_at = expires_at
        self.rate_limit_rpm = rate_limit_rpm
        self.created_at = datetime.now(timezone.utc)
        self.last_used_at: Optional[datetime] = None

    def verify(self, secret: str) -> bool:
        """验证 API Key"""
        return hmac.compare_digest(
            hashlib.sha256(secret.encode()).hexdigest(),
            self.secret_hash,
        )

    @property
    def is_expired(self) -> bool:
        if not self.expires_at:
            return False
        return datetime.now(timezone.utc) > self.expires_at


class APIKeyStore:
    """API Key 存储（生产环境应使用数据库）"""

    def __init__(self):
        self._keys: dict[str, APIKey] = {}

    def add_key(self, key: APIKey):
        self._keys[key.secret_hash] = key

    def verify(self, secret: str) -> Optional[APIKey]:
        """验证并返回 API Key 信息"""
        key_hash = hashlib.sha256(secret.encode()).hexdigest()
        api_key = self._keys.get(key_hash)

        if not api_key:
            return None
        if api_key.is_expired:
            return None

        api_key.last_used_at = datetime.now(timezone.utc)
        return api_key


async def api_key_middleware(request, call_next):
    """FastAPI middleware"""
    from fastapi import HTTPException

    # 白名单路径（无需认证）
    public_paths = ["/health", "/ready", "/metrics", "/docs", "/openapi.json"]
    if any(request.url.path.startswith(p) for p in public_paths):
        return await call_next(request)

    api_key_header = request.headers.get("X-API-Key")
    if not api_key_header:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")

    key_store = get_api_key_store()  # 从依赖注入获取
    api_key = key_store.verify(api_key_header)

    if not api_key:
        raise HTTPException(status_code=403, detail="Invalid or expired API key")

    # 将认证信息注入 request state
    request.state.api_key = api_key
    return await call_next(request)


def get_api_key_store() -> APIKeyStore:
    """获取全局 API Key Store"""
    import config.manager as cm
    manager = cm.get_manager()
    if not hasattr(manager, "_api_key_store"):
        manager._api_key_store = APIKeyStore()
        # 从配置加载 keys
        for key_config in (manager.settings.security or {}).get("api_keys", []):
            expires_at = None
            if key_config.get("expires_at"):
                expires_at = datetime.fromisoformat(key_config["expires_at"])

            manager._api_key_store.add_key(APIKey(
                name=key_config["name"],
                secret=key_config["secret"],
                roles=key_config.get("roles", ["user"]),
                expires_at=expires_at,
                rate_limit_rpm=key_config.get("rate_limit_rpm", 60),
            ))
    return manager._api_key_store
```

---

## 3. RBAC 权限控制 `security/rbac.py`

```python
"""
RBAC（Role-Based Access Control）权限模型。

角色层级：
- admin: 所有权限，可管理用户和配置
- developer: 可使用所有工具，可访问审计日志
- user: 基础对话 + 有限工具集
- viewer: 只读访问

资源操作矩阵：
| 资源          | admin | developer | user | viewer |
|---------------|-------|-----------|------|--------|
| chat          | ✓     | ✓         | ✓    | ✓      |
| tool_call     | ✓     | ✓         | partial| ✗   |
| config_read   | ✓     | ✓         | ✗    | ✗      |
| config_write  | ✓     | ✗         | ✗    | ✗      |
| session_mgmt  | ✓     | ✓         | own  | ✗      |
| audit_logs    | ✓     | ✓         | ✗    | ✗      |
"""

from enum import Enum
from typing import Set


class Permission(str, Enum):
    CHAT = "chat"                    # 发起对话
    TOOL_CALL = "tool_call"          # 调用工具
    TOOL_CALL_DANGEROUS = "tool_call.dangerous"  # 调用危险工具（代码执行、DB写操作）
    CONFIG_READ = "config.read"      # 读取配置
    CONFIG_WRITE = "config.write"    # 修改配置
    SESSION_OWN = "session.own"      # 管理自己的会话
    SESSION_ALL = "session.all"      # 管理所有会话
    AUDIT_LOGS = "audit.logs"        # 查看审计日志
    RAG_ACCESS = "rag.access"        # 访问知识库


class Role:
    """角色定义"""

    def __init__(self, name: str, permissions: Set[Permission]):
        self.name = name
        self.permissions = permissions

    def has_permission(self, permission: Permission) -> bool:
        return permission in self.permissions


# 预定义角色
ROLES = {
    "admin": Role("admin", set(Permission)),
    "developer": Role("developer", {
        Permission.CHAT, Permission.TOOL_CALL, Permission.CONFIG_READ,
        Permission.SESSION_ALL, Permission.AUDIT_LOGS, Permission.RAG_ACCESS,
    }),
    "user": Role("user", {
        Permission.CHAT, Permission.TOOL_CALL, Permission.SESSION_OWN,
    }),
    "viewer": Role("viewer", {
        Permission.CHAT,
    }),
}


class PermissionChecker:
    """权限检查器"""

    def __init__(self):
        self._role_overrides: dict[str, Set[Permission]] = {}

    def check(self, user_roles: list[str], permission: Permission) -> bool:
        """检查用户是否有某项权限"""
        for role_name in user_roles:
            role = ROLES.get(role_name)
            if role and role.has_permission(permission):
                return True
        return False

    def filter_tools(self, tools: list[dict], user_roles: list[str]) -> list[dict]:
        """根据权限过滤可用工具"""
        can_dangerous = self.check(user_roles, Permission.TOOL_CALL_DANGEROUS)

        filtered = []
        for tool in tools:
            if tool.get("dangerous", False) and not can_dangerous:
                continue
            filtered.append(tool)
        return filtered


def get_permission_checker() -> PermissionChecker:
    """获取全局权限检查器"""
    import config.manager as cm
    manager = cm.get_manager()
    if not hasattr(manager, "_permission_checker"):
        manager._permission_checker = PermissionChecker()
    return manager._permission_checker
```

---

## 4. 内容过滤 `security/content_filter.py`

```python
"""
输入输出内容过滤器。

检测类型：
- 敏感词过滤（关键词匹配）
- PII 检测（个人身份信息脱敏）
- Prompt 注入检测
- 输出安全审查
"""

import re
from typing import List, Optional


class ContentFilter:
    """内容过滤器"""

    # 常见 prompt injection 模式
    INJECTION_PATTERNS = [
        r"(?i)ignore\s+previous\s+instructions",
        r"(?i)disregard\s+(the\s+)?system\s*(prompt|message)",
        r"(?i)you\s+are\s+now\s+called",
        r"(?i)forget\s+your\s+(instructions|rules)",
        r"(?i)repeat\s+the\s+above",
        r"(?i)output\s+(the|your)\s*(system|full)\s*prompt",
    ]

    # PII 模式（简化版）
    PII_PATTERNS = {
        "email": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
        "phone_cn": re.compile(r"1[3-9]\d{9}"),
        "id_card": re.compile(r"\d{17}[\dXx]"),
        "credit_card": re.compile(r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b"),
    }

    def __init__(self, sensitive_words: List[str] = None):
        self.sensitive_words = set(sensitive_words or [])
        self._injection_regexes = [re.compile(p) for p in self.INJECTION_PATTERNS]

    def check_injection(self, text: str) -> bool:
        """检测 prompt 注入攻击"""
        return any(rx.search(text) for rx in self._injection_regexes)

    def detect_pii(self, text: str) -> List[dict]:
        """检测 PII（个人身份信息）"""
        findings = []
        for pii_type, pattern in self.PII_PATTERNS.items():
            for match in pattern.finditer(text):
                findings.append({
                    "type": pii_type,
                    "start": match.start(),
                    "end": match.end(),
                    "value": "***" + match.group()[-4:],  # 部分遮蔽
                })
        return findings

    def mask_pii(self, text: str) -> str:
        """脱敏 PII"""
        masked = text
        for pii_type, pattern in self.PII_PATTERNS.items():
            masked = pattern.sub("[REDACTED]", masked)
        return masked

    def check_sensitive_words(self, text: str) -> List[str]:
        """检测敏感词"""
        found = []
        text_lower = text.lower()
        for word in self.sensitive_words:
            if word.lower() in text_lower:
                found.append(word)
        return found

    def filter_input(self, text: str) -> dict:
        """
        输入过滤，返回检查结果。

        Returns:
            {
                "allowed": True/False,
                "reasons": ["injection_detected", ...],
                "masked_text": "...",
                "pii_found": [...],
            }
        """
        reasons = []
        masked_text = text

        if self.check_injection(text):
            reasons.append("prompt_injection_detected")

        pii_findings = self.detect_pii(text)
        if pii_findings:
            reasons.append(f"pii_found ({len(pii_findings)} items)")
            masked_text = self.mask_pii(text)

        sensitive = self.check_sensitive_words(text)
        if sensitive:
            reasons.append(f"sensitive_words ({', '.join(sensitive)})")

        return {
            "allowed": len(reasons) == 0,
            "reasons": reasons,
            "masked_text": masked_text,
            "pii_found": pii_findings,
        }


class OutputFilter:
    """输出过滤器（审查 LLM 回复）"""

    def __init__(self, content_filter: ContentFilter):
        self._filter = content_filter

    async def check(self, text: str) -> dict:
        """检查 LLM 输出是否安全"""
        pii_findings = self._filter.detect_pii(text)
        sensitive = self._filter.check_sensitive_words(text)

        return {
            "safe": len(pii_findings) == 0 and len(sensitive) == 0,
            "pii_count": len(pii_findings),
            "sensitive_words": sensitive,
        }
```

---

## 5. 速率限制 `security/rate_limiter.py`

```python
"""
速率限制器 —— 按 API key / IP 限流。

算法：滑动窗口计数器（Sliding Window Counter）
"""

import time
from collections import defaultdict
from typing import Optional


class RateLimitEntry:
    """单个窗口的请求计数"""

    def __init__(self, window_start: float, count: int = 0):
        self.window_start = window_start
        self.count = count


class RateLimiter:
    """滑动窗口速率限制器"""

    def __init__(self, default_rpm: int = 60, window_seconds: float = 60.0):
        self.default_rpm = default_rpm
        self.window_seconds = window_seconds
        self._requests: dict[str, list[RateLimitEntry]] = defaultdict(list)

    def is_allowed(self, key: str, limit: Optional[int] = None) -> tuple[bool, dict]:
        """
        检查请求是否允许。

        Returns:
            (allowed, info_dict)
            info_dict: {"remaining": N, "limit": N, "retry_after": seconds}
        """
        effective_limit = limit or self.default_rpm
        now = time.time()
        window_start = now - self.window_seconds

        # 清理过期窗口
        entries = self._requests[key]
        self._requests[key] = [e for e in entries if e.window_start > window_start]
        entries = self._requests[key]

        # 计算当前窗口总计数
        total_count = sum(e.count for e in entries)

        if total_count >= effective_limit:
            # 找到最早窗口的过期时间
            oldest = min(entries, key=lambda e: e.window_start)
            retry_after = oldest.window_start + self.window_seconds - now
            return False, {
                "remaining": 0,
                "limit": effective_limit,
                "retry_after": max(0.1, retry_after),
            }

        # 添加当前请求到最新窗口
        current_window = None
        for e in entries:
            if abs(e.window_start - now) < 1.0:
                current_window = e
                break

        if current_window:
            current_window.count += 1
        else:
            entries.append(RateLimitEntry(window_start=now, count=1))

        return True, {
            "remaining": effective_limit - total_count - 1,
            "limit": effective_limit,
            "retry_after": 0,
        }

    def reset(self, key: str):
        """重置指定 key 的计数"""
        self._requests.pop(key, None)


def get_rate_limiter() -> RateLimiter:
    """获取全局速率限制器"""
    import config.manager as cm
    manager = cm.get_manager()
    if not hasattr(manager, "_rate_limiter"):
        settings = (manager.settings.security or {}).get("rate_limit", {})
        manager._rate_limiter = RateLimiter(
            default_rpm=settings.get("default_rpm", 60),
            window_seconds=settings.get("window_seconds", 60.0),
        )
    return manager._rate_limiter


async def rate_limit_middleware(request, call_next):
    """FastAPI middleware"""
    from fastapi import HTTPException

    public_paths = ["/health", "/ready", "/metrics"]
    if any(request.url.path.startswith(p) for p in public_paths):
        return await call_next(request)

    rate_limiter = get_rate_limiter()

    # 优先使用 API key，否则使用 IP
    limit_key = None
    rpm_limit = None

    if hasattr(request.state, "api_key"):
        limit_key = request.state.api_key.name
        rpm_limit = request.state.api_key.rate_limit_rpm
    else:
        limit_key = request.client.host

    allowed, info = rate_limiter.is_allowed(limit_key, rpm_limit)

    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded",
            headers={
                "Retry-After": str(int(info["retry_after"]) + 1),
                "X-RateLimit-Limit": str(info["limit"]),
                "X-RateLimit-Remaining": "0",
            },
        )

    response = await call_next(request)
    response.headers["X-RateLimit-Limit"] = str(info["limit"])
    response.headers["X-RateLimit-Remaining"] = str(info["remaining"])
    return response
```

---

## 6. 审计日志 `security/audit.py`

```python
"""
操作审计日志。

记录所有关键操作，用于合规审查和安全分析。
"""

import json
from datetime import datetime, timezone


class AuditLogger:
    """审计日志记录器"""

    def __init__(self):
        self._logger = None  # 使用 observability/logging.py 的 JSON logger

    def log(
        self, action: str, user_id: str = None, api_key_name: str = None,
        resource: str = None, detail: dict = None, success: bool = True,
    ):
        """记录审计事件"""
        from observability.logging import setup_logging

        if not self._logger:
            self._logger = __import__("logging").getLogger("audit")

        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "user_id": user_id,
            "api_key_name": api_key_name,
            "resource": resource,
            "success": success,
            "detail": detail or {},
        }

        self._logger.info(
            json.dumps(log_entry, ensure_ascii=False),
            extra={"module": "audit"},
        )


def get_audit_logger() -> AuditLogger:
    """获取全局审计日志记录器"""
    import config.manager as cm
    manager = cm.get_manager()
    if not hasattr(manager, "_audit_logger"):
        manager._audit_logger = AuditLogger()
    return manager._audit_logger
```

---

## 7. YAML 配置 `security` section

```yaml
# config/settings.yaml (新增)
security:
  api_keys:
    - name: "admin-key"
      secret: "${ADMIN_API_KEY}"       # 从环境变量读取
      roles: ["admin"]
      rate_limit_rpm: 120
      expires_at: ""                   # ISO8601，空 = 永不过期

    - name: "dev-key"
      secret: "${DEV_API_KEY}"
      roles: ["developer"]
      rate_limit_rpm: 60

  rate_limit:
    default_rpm: 30                    # 未认证请求的默认限流
    window_seconds: 60.0

  content_filter:
    enabled: true
    sensitive_words: []                # 自定义敏感词列表
    pii_detection: true
    injection_detection: true

  rbac:
    default_role: "user"              # 未指定角色时的默认值
```

---

## 8. 设计总结

| 特性 | 实现方式 |
|------|---------|
| **API Key 认证** | SHA-256 hash + HMAC compare，支持过期控制 |
| **RBAC** | 4 级角色（admin/developer/user/viewer），权限矩阵过滤工具 |
| **内容过滤** | Prompt 注入检测、PII 脱敏、敏感词拦截 |
| **速率限制** | 滑动窗口计数器，按 API key / IP 限流 |
| **审计日志** | JSON 格式记录所有关键操作 |

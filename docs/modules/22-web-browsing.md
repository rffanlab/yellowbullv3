# Web 浏览工具详细设计（Web Browsing）

## 1. 职责边界

| 职责 | 说明 |
|------|------|
| **网页访问** | 发起 HTTP 请求获取网页内容，支持 GET/POST |
| **内容提取** | 从 HTML 中提取结构化信息（标题、正文、链接等） |
| **智能搜索** | 调用搜索引擎 API，解析搜索结果 |
| **页面交互** | 通过无头浏览器模拟用户操作（点击、填写表单、滚动） |
| **截图与可视化** | 对网页进行截图，支持视觉分析 |

---

## 2. 协议设计 `tools/web/protocol.py`

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ActionType(str, Enum):
    CLICK = "click"               # 点击元素
    TYPE = "type"                 # 输入文本
    SCROLL = "scroll"             # 滚动页面
    WAIT = "wait"                 # 等待指定时间
    GOTO = "goto"                 # 导航到 URL
    SCREENSHOT = "screenshot"     # 截图
    EXTRACT = "extract"           # 提取内容


@dataclass(frozen=True)
class BrowserAction:
    """浏览器操作指令"""
    action_type: ActionType       # 操作类型
    selector: str | None = None   # CSS 选择器或 XPath
    value: str | None = None      # 输入值（TYPE 操作）
    duration_ms: int | None = None  # 等待时长（WAIT 操作）
    url: str | None = None        # 目标 URL（GOTO 操作)


@dataclass(frozen=True)
class WebPageContent:
    """网页内容"""
    url: str                      # 页面 URL
    title: str                    # 页面标题
    text_content: str             # 纯文本内容
    html_content: str | None = None   # 原始 HTML（可选）
    links: list[dict[str, str]] = field(default_factory=list)  # 页面链接 [{"text": "...", "href": "..."}]
    images: list[dict[str, str]] = field(default_factory=list)  # 图片信息 [{"alt": "...", "src": "..."}]
    metadata: dict[str, str] = field(default_factory=dict)      # 页面元数据


@dataclass(frozen=True)
class SearchResult:
    """搜索结果"""
    query: str                    # 搜索关键词
    results: list["SearchResultItem"]  # 结果列表
    total_count: int | None = None  # 总结果数


@dataclass(frozen=True)
class SearchResultItem:
    """单个搜索结果"""
    title: str                    # 标题
    url: str                      # 链接
    snippet: str                  # 摘要
    position: int                 # 排名位置


@dataclass(frozen=True)
class BrowserState:
    """浏览器状态快照"""
    current_url: str              # 当前 URL
    title: str                    # 页面标题
    screenshot_base64: str | None = None  # 截图（base64）
    interactive_elements: list[dict[str, Any]] = field(default_factory=list)  # 可交互元素列表
```

---

## 3. HTTP 网页抓取 `tools/web/http_fetcher.py`

```python
"""
轻量级 HTTP 网页抓取器。

适用于：静态页面、API 端点、RSS/Atom feed
不适用：需要 JavaScript 渲染的动态页面
"""

import asyncio
import logging
import re
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

import aiohttp

logger = logging.getLogger(__name__)


class HTMLTextExtractor(HTMLParser):
    """从 HTML 中提取纯文本内容"""

    SKIP_TAGS = {"script", "style", "noscript", "iframe", "svg"}

    def __init__(self):
        super().__init__()
        self._text_parts: list[str] = []
        self._skip = False
        self._title = ""
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self.SKIP_TAGS:
            self._skip = True
        if tag == "title":
            self._in_title = True

        # 添加换行
        block_tags = {"p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "br"}
        if tag in block_tags and not self._skip:
            self._text_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIP_TAGS:
            self._skip = False
        if tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        if self._in_title:
            self._title = data.strip()
        else:
            self._text_parts.append(data)

    @property
    def title(self) -> str:
        return self._title

    @property
    def text(self) -> str:
        return " ".join("".join(self._text_parts).split())


class HTTPFetcher:
    """
    HTTP 网页抓取器。

    特性：
    - 支持自定义 User-Agent、Headers
    - 自动处理重定向
    - 超时控制与重试机制
    - HTML → 纯文本转换
    """

    DEFAULT_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    }

    def __init__(
        self,
        timeout: float = 15.0,
        max_retries: int = 2,
        max_content_length: int = 1_000_000,  # 1MB
    ):
        self._timeout = aiohttp.ClientTimeout(total=timeout)
        self._max_retries = max_retries
        self._max_content_length = max_content_length

    async def fetch(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        extract_links: bool = True,
    ) -> WebPageContent:
        """
        抓取网页内容。

        Args:
            url:           目标 URL
            headers:       自定义请求头（会合并默认头）
            extract_links: 是否提取页面链接

        Returns:
            WebPageContent
        """
        merged_headers = {**self.DEFAULT_HEADERS, **(headers or {})}

        last_error = None
        for attempt in range(self._max_retries + 1):
            try:
                async with aiohttp.ClientSession(timeout=self._timeout) as session:
                    async with session.get(url, headers=merged_headers) as resp:
                        resp.raise_for_status()
                        html = await resp.text(encoding="utf-8")

                        # 截断过长内容
                        if len(html) > self._max_content_length:
                            html = html[:self._max_content_length]

                        return self._parse_html(url, html, extract_links)

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_error = e
                logger.warning(
                    f"Fetch attempt {attempt + 1} failed for {url}: {e}"
                )
                if attempt < self._max_retries:
                    await asyncio.sleep(0.5 * (2 ** attempt))

        raise RuntimeError(f"Failed to fetch {url} after {self._max_retries + 1} attempts: {last_error}")

    def _parse_html(
        self, url: str, html: str, extract_links: bool
    ) -> WebPageContent:
        """解析 HTML，提取结构化内容"""
        extractor = HTMLTextExtractor()
        try:
            extractor.feed(html)
        except Exception:
            # 解析失败时返回原始文本
            clean_text = re.sub(r"<[^>]+>", "", html)
            return WebPageContent(
                url=url,
                title="",
                text_content=clean_text[:50000],
            )

        links = []
        if extract_links:
            link_pattern = re.findall(
                r'<a\s+[^>]*href=["\']([^"\']+)["\'][^>]*>([^<]*)</a>',
                html,
                re.IGNORECASE | re.DOTALL,
            )
            for href, text in link_pattern[:50]:  # 最多提取 50 个链接
                absolute_url = urljoin(url, href)
                links.append({"text": text.strip(), "href": absolute_url})

        return WebPageContent(
            url=url,
            title=extractor.title,
            text_content=extractor.text[:50000],  # 限制长度
            links=links,
        )
```

---

## 4. 无头浏览器 `tools/web/headless_browser.py`

```python
"""
无头浏览器控制器。

基于 Playwright，支持：
- JavaScript 渲染页面
- 模拟用户交互（点击、输入、滚动）
- 截图与 PDF 导出
- 表单填写与提交
"""

import asyncio
import base64
import logging
from typing import Any, Optional

from playwright.async_api import async_playwright, Browser as PlaywrightBrowser, Page

logger = logging.getLogger(__name__)


class HeadlessBrowser:
    """
    无头浏览器管理器。

    使用单例模式管理浏览器实例，避免频繁启动/关闭的开销。
    """

    _instance: Optional["HeadlessBrowser"] = None
    _playwright = None
    _browser: Optional[PlaywrightBrowser] = None

    def __new__(cls) -> "HeadlessBrowser":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized"):
            return
        self._initialized = True
        self._pages: dict[str, Page] = {}  # session_id -> Page
        self._viewport = {"width": 1280, "height": 900}

    async def start(self) -> None:
        """启动浏览器实例"""
        if self._browser is not None:
            return

        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        )
        logger.info("Headless browser started")

    async def stop(self) -> None:
        """关闭浏览器实例"""
        for page in self._pages.values():
            await page.close()
        self._pages.clear()

        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()

        self._browser = None
        self._playwright = None
        logger.info("Headless browser stopped")

    async def get_page(self, session_id: str) -> Page:
        """获取或创建浏览器页面"""
        if session_id not in self._pages:
            context = await self._browser.new_context(
                viewport=self._viewport,
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
                ),
            )
            self._pages[session_id] = await context.new_page()

        return self._pages[session_id]

    async def navigate(self, session_id: str, url: str) -> BrowserState:
        """导航到指定 URL"""
        page = await self.get_page(session_id)
        await page.goto(url, wait_until="networkidle", timeout=30000)
        return await self._get_state(page)

    async def click(self, session_id: str, selector: str) -> BrowserState:
        """点击元素"""
        page = await self.get_page(session_id)
        await page.click(selector)
        await asyncio.sleep(0.5)  # 等待页面响应
        return await self._get_state(page)

    async def type_text(self, session_id: str, selector: str, text: str) -> BrowserState:
        """在输入框中输入文本"""
        page = await self.get_page(session_id)
        await page.fill(selector, text)
        return await self._get_state(page)

    async def scroll(
        self, session_id: str, direction: str = "down", amount: int = 500
    ) -> BrowserState:
        """滚动页面"""
        page = await self.get_page(session_id)
        delta = amount if direction == "down" else -amount
        await page.evaluate(f"window.scrollBy(0, {delta})")
        await asyncio.sleep(0.3)
        return await self._get_state(page)

    async def screenshot(self, session_id: str) -> str:
        """截取页面截图，返回 base64"""
        page = await self.get_page(session_id)
        img_bytes = await page.screenshot(full_page=False)
        return base64.b64encode(img_bytes).decode("utf-8")

    async def extract_content(self, session_id: str) -> WebPageContent:
        """提取当前页面内容"""
        page = await self.get_page(session_id)
        url = page.url
        title = await page.title()

        # 提取正文文本（排除导航、侧栏等）
        text_content = await page.evaluate("""
            () => {
                const body = document.body.cloneNode(true);
                // 移除脚本和样式
                body.querySelectorAll('script, style, noscript, iframe').forEach(el => el.remove());
                return body.innerText.trim();
            }
        """)

        # 提取链接
        links = await page.evaluate("""
            () => {
                return Array.from(document.querySelectorAll('a[href]')).slice(0, 50).map(a => ({
                    text: a.textContent.trim(),
                    href: a.href
                })).filter(l => l.text.length > 0);
            }
        """)

        return WebPageContent(
            url=url,
            title=title,
            text_content=text_content[:50000],
            links=links,
        )

    async def execute_actions(
        self, session_id: str, actions: list[dict[str, Any]]
    ) -> BrowserState:
        """
        执行一系列浏览器操作。

        Args:
            session_id: 会话 ID
            actions:    操作列表，每项为 BrowserAction 的字典表示

        Returns:
            最终浏览器状态
        """
        state = None
        for action_dict in actions:
            action = BrowserAction(**action_dict)

            if action.action_type == ActionType.GOTO:
                state = await self.navigate(session_id, action.url or "")
            elif action.action_type == ActionType.CLICK:
                state = await self.click(session_id, action.selector or "")
            elif action.action_type == ActionType.TYPE:
                state = await self.type_text(
                    session_id, action.selector or "", action.value or ""
                )
            elif action.action_type == ActionType.SCROLL:
                state = await self.scroll(session_id)
            elif action.action_type == ActionType.WAIT:
                await asyncio.sleep((action.duration_ms or 1000) / 1000)
                state = await self._get_state(await self.get_page(session_id))
            elif action.action_type == ActionType.SCREENSHOT:
                await self.screenshot(session_id)
                state = await self._get_state(await self.get_page(session_id))

        return state or await self._get_state(await self.get_page(session_id))

    async def _get_state(self, page: Page) -> BrowserState:
        """获取浏览器状态快照"""
        elements = await page.evaluate("""
            () => {
                return Array.from(document.querySelectorAll('a, button, input, select, textarea')).slice(0, 30).map(el => ({
                    tag: el.tagName.toLowerCase(),
                    text: (el.textContent || '').trim().substring(0, 100),
                    id: el.id,
                    type: el.type || null,
                }));
            }
        """)

        return BrowserState(
            current_url=page.url,
            title=await page.title(),
            interactive_elements=elements,
        )
```

---

## 5. Web 搜索工具 `tools/web/search.py`

```python
"""
Web 搜索引擎集成。

支持多个搜索引擎后端：
- Google Custom Search API
- Bing Search API
- DuckDuckGo（免费，无需 API key）
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any

import aiohttp

from tools.web.protocol import SearchResult, SearchResultItem

logger = logging.getLogger(__name__)


class BaseSearchEngine(ABC):
    """搜索引擎基类"""

    @property
    @abstractmethod
    def engine_name(self) -> str:
        ...

    @abstractmethod
    async def search(
        self, query: str, num_results: int = 10
    ) -> SearchResult:
        ...


class DuckDuckGoSearch(BaseSearchEngine):
    """
    DuckDuckGo 搜索引擎。

    优势：免费、无需 API key、尊重隐私
    劣势：结果质量不如 Google/Bing，可能有速率限制
    """

    @property
    def engine_name(self) -> str:
        return "duckduckgo"

    async def search(
        self, query: str, num_results: int = 10
    ) -> SearchResult:
        """使用 DuckDuckGo HTML 搜索（非官方 API）"""
        url = "https://html.duckduckgo.com/html/"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
            )
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, headers=headers, data={"q": query}
                ) as resp:
                    html = await resp.text()

                    # 解析搜索结果
                    results = self._parse_results(html)

                    return SearchResult(
                        query=query,
                        results=results[:num_results],
                    )

        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}")
            return SearchResult(query=query, results=[])

    def _parse_results(self, html: str) -> list[SearchResultItem]:
        """解析 DuckDuckGo HTML 搜索结果"""
        import re

        results = []
        # 匹配结果块
        blocks = re.findall(
            r'<a\s+class="result__a"\s+href="([^"]*)"[^>]*>([^<]*)</a>'
            r'.*?<a\s+class="result__snippet"[^>]*>([^<]*)</a>',
            html,
            re.DOTALL,
        )

        for i, (url, title, snippet) in enumerate(blocks):
            results.append(SearchResultItem(
                title=re.sub(r"<[^>]+>", "", title).strip(),
                url=url,
                snippet=re.sub(r"<[^>]+>", "", snippet).strip(),
                position=i + 1,
            ))

        return results


class SearchEngineManager:
    """
    搜索引擎管理器。

    支持多引擎 fallback，优先使用配置的主引擎。
    """

    def __init__(self):
        self._engines: list[BaseSearchEngine] = [DuckDuckGoSearch()]

    async def search(
        self, query: str, num_results: int = 10
    ) -> SearchResult:
        """执行搜索，自动 fallback"""
        for engine in self._engines:
            try:
                result = await engine.search(query, num_results)
                if result.results:
                    return result
            except Exception as e:
                logger.warning(f"Search engine {engine.engine_name} failed: {e}")

        return SearchResult(query=query, results=[])
```

---

## 6. Web 浏览工具注册 `tools/web/browsing_tool.py`

```python
"""
Web 浏览工具，注册到 ToolRegistry。

提供统一的 web_browse 工具接口，Agent 可通过自然语言调用。
"""

import json
import logging
from typing import Any

from tools.base import BaseTool, ToolParameter, ToolSchema
from tools.web.http_fetcher import HTTPFetcher, WebPageContent
from tools.web.headless_browser import HeadlessBrowser
from tools.web.search import SearchEngineManager

logger = logging.getLogger(__name__)


class WebBrowseTool(BaseTool):
    """
    Web 浏览工具。

    支持的操作：
    - search: 搜索关键词，返回结果列表
    - fetch: 抓取指定 URL 的网页内容
    - browse: 使用无头浏览器访问页面（支持 JS 渲染）
    - interact: 执行一系列浏览器操作
    """

    NAME = "web_browse"
    DESCRIPTION = (
        "浏览 Web 页面。支持搜索、抓取网页内容、"
        "模拟用户交互等操作。"
    )

    SCHEMA = ToolSchema(
        name=NAME,
        description=DESCRIPTION,
        parameters=[
            ToolParameter(
                name="operation",
                type="string",
                enum=["search", "fetch", "browse", "interact"],
                required=True,
                description="操作类型：search 搜索, fetch 抓取页面, browse 浏览器访问, interact 交互操作",
            ),
            ToolParameter(
                name="query",
                type="string",
                required=False,
                description="搜索关键词（operation=search 时必填）",
            ),
            ToolParameter(
                name="url",
                type="string",
                required=False,
                description="目标 URL（operation=fetch/browse 时必填）",
            ),
            ToolParameter(
                name="actions",
                type="array",
                required=False,
                description="浏览器操作序列（operation=interact 时使用）",
            ),
            ToolParameter(
                name="max_results",
                type="integer",
                required=False,
                default=5,
                description="搜索结果数量上限",
            ),
        ],
    )

    def __init__(self):
        self._fetcher = HTTPFetcher()
        self._browser = HeadlessBrowser()
        self._search_manager = SearchEngineManager()
        self._started = False

    async def start(self) -> None:
        """启动浏览器（懒加载）"""
        if not self._started:
            await self._browser.start()
            self._started = True

    async def execute(self, arguments: dict[str, Any]) -> str:
        operation = arguments.get("operation", "fetch")

        if operation == "search":
            return await self._handle_search(arguments)
        elif operation == "fetch":
            return await self._handle_fetch(arguments)
        elif operation == "browse":
            await self.start()
            return await self._handle_browse(arguments)
        elif operation == "interact":
            await self.start()
            return await self._handle_interact(arguments)
        else:
            return f"Unknown operation: {operation}"

    async def _handle_search(self, args: dict[str, Any]) -> str:
        """处理搜索操作"""
        query = args.get("query", "")
        max_results = args.get("max_results", 5)

        if not query:
            return "Error: 'query' is required for search operation."

        result = await self._search_manager.search(query, max_results)

        if not result.results:
            return f"No results found for '{query}'."

        lines = [f"搜索结果（共 {len(result.results)} 条）：\n"]
        for i, item in enumerate(result.results):
            lines.append(f"{i + 1}. [{item.title}]({item.url})")
            lines.append(f"   {item.snippet[:200]}")
            lines.append("")

        return "\n".join(lines)

    async def _handle_fetch(self, args: dict[str, Any]) -> str:
        """处理页面抓取"""
        url = args.get("url", "")

        if not url:
            return "Error: 'url' is required for fetch operation."

        try:
            content = await self._fetcher.fetch(url)
            return (
                f"## {content.title}\n"
                f"**URL**: {content.url}\n\n"
                f"{content.text_content[:8000]}"
            )
        except Exception as e:
            return f"Failed to fetch page: {e}"

    async def _handle_browse(self, args: dict[str, Any]) -> str:
        """处理浏览器访问"""
        url = args.get("url", "")
        session_id = args.get("session_id", "default")

        if not url:
            return "Error: 'url' is required for browse operation."

        try:
            state = await self._browser.navigate(session_id, url)
            content = await self._browser.extract_content(session_id)

            return (
                f"## {content.title}\n"
                f"**URL**: {state.current_url}\n\n"
                f"{content.text_content[:8000]}"
            )
        except Exception as e:
            return f"Failed to browse page: {e}"

    async def _handle_interact(self, args: dict[str, Any]) -> str:
        """处理浏览器交互"""
        session_id = args.get("session_id", "default")
        actions = args.get("actions", [])

        if not actions:
            return "Error: 'actions' is required for interact operation."

        try:
            state = await self._browser.execute_actions(session_id, actions)
            content = await self._browser.extract_content(session_id)

            return (
                f"## {content.title}\n"
                f"**URL**: {state.current_url}\n\n"
                f"{content.text_content[:8000]}"
            )
        except Exception as e:
            return f"Failed to interact with page: {e}"
```

---

## 7. 安全限制 `tools/web/security.py`

```python
"""
Web 浏览安全控制。

防止：
- 访问内网地址
- 过度请求（速率限制）
- 恶意内容下载
"""

import ipaddress
import logging
import time
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class WebSecurityPolicy:
    """Web 浏览安全策略"""

    # 禁止访问的内网 IP 段
    PRIVATE_NETWORKS = [
        ipaddress.ip_network("10.0.0.0/8"),
        ipaddress.ip_network("172.16.0.0/12"),
        ipaddress.ip_network("192.168.0.0/16"),
        ipaddress.ip_network("127.0.0.0/8"),
        ipaddress.ip_network("0.0.0.0/8"),
    ]

    # 禁止的文件类型
    BLOCKED_MIME_TYPES = {
        "application/x-executable",
        "application/x-msdownload",
        "application/octet-stream",
    }

    def __init__(
        self,
        max_requests_per_minute: int = 30,
        max_page_size_bytes: int = 5_000_000,  # 5MB
    ):
        self._max_rpm = max_requests_per_minute
        self._max_page_size = max_page_size_bytes
        self._request_times: list[float] = []

    def is_url_allowed(self, url: str) -> tuple[bool, str]:
        """检查 URL 是否允许访问"""
        parsed = urlparse(url)

        # 只允许 http/https
        if parsed.scheme not in ("http", "https"):
            return False, f"Only HTTP/HTTPS URLs are allowed, got: {parsed.scheme}"

        # 检查是否为内网地址
        hostname = parsed.hostname or ""
        if self._is_private_host(hostname):
            return False, f"Access to private/internal network is blocked: {hostname}"

        return True, "OK"

    def _is_private_host(self, hostname: str) -> bool:
        """检查主机是否为内网地址"""
        # localhost 直接拒绝
        if hostname.lower() in ("localhost", "local", "internal"):
            return True

        try:
            ip = ipaddress.ip_address(hostname)
            return any(ip in net for net in self.PRIVATE_NETWORKS)
        except ValueError:
            # 域名，不做 IP 检查（DNS 可能在运行时解析为内网）
            return False

    def check_rate_limit(self) -> bool:
        """检查是否超过速率限制"""
        now = time.time()
        # 清理一分钟前的记录
        self._request_times = [t for t in self._request_times if now - t < 60]

        if len(self._request_times) >= self._max_rpm:
            logger.warning("Web browsing rate limit exceeded")
            return False

        self._request_times.append(now)
        return True
```

---

## 8. 配置项 `config/web_browsing.yaml`

```yaml
web_browsing:
  http_fetcher:
    timeout_seconds: 15
    max_retries: 2
    max_content_length_bytes: 1_000_000   # 1MB

  headless_browser:
    enabled: false                        # 默认不启用（需要额外依赖）
    viewport_width: 1280
    viewport_height: 900
    max_concurrent_pages: 5               # 最大并发页面数

  search:
    primary_engine: duckduckgo            # duckduckgo | google | bing
    api_key: ""                           # Google/Bing API key（DuckDuckGo 不需要）
    max_results: 10

  security:
    max_requests_per_minute: 30           # 每分钟最大请求数
    max_page_size_bytes: 5_000_000       # 单页最大大小
    block_private_networks: true          # 阻止内网访问
```

---

## 9. 与现有模块的交互

| 模块 | 交互方式 |
|------|---------|
| `03-tool-system` | WebBrowseTool 注册到 ToolRegistry |
| `05-tool-executor` | 通过 ToolExecutor 执行 web_browse 工具调用 |
| `14-code-execution` | 代码沙箱中可限制网络访问权限 |
| `10-security-governance` | URL 过滤、内容安全检查 |

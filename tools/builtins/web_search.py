"""Web search tool."""

from typing import Any

from tools.base import BaseTool, ToolInfo, ToolResult
from tools.registry import register_tool


@register_tool(
    name="web_search",
    description="Search the internet for up-to-date information. Use for news, facts, and real-time queries.",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query keywords",
            },
            "max_results": {
                "type": "integer",
                "description": "Number of results to return, default 5",
            },
        },
        "required": ["query"],
    },
)
class WebSearchTool(BaseTool):
    def __init__(self):
        self.engine = "duckduckgo"

    @property
    def info(self) -> ToolInfo:
        return self._info  # type: ignore[attr-defined]

    async def execute(self, query: str, max_results: int = 5) -> ToolResult:
        try:
            from duckduckgo_search import DDGS

            results = DDGS().text(query, max_results=max_results)
            formatted = "\n\n".join(
                f"[{r.get('title', 'N/A')}]({r.get('href', '')})\n"
                f"{r.get('body', r.get('description', ''))}"
                for r in results[:max_results]
            )
            return ToolResult(content=formatted or "No results found")
        except ImportError:
            return ToolResult(
                content="Web search unavailable: duckduckgo_search not installed",
                success=False,
            )
        except Exception as e:
            return ToolResult(content=f"Search failed: {e}", success=False)

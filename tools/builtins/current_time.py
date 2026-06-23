"""Current time tool."""

from datetime import UTC, datetime

from tools.base import BaseTool, ToolInfo, ToolResult
from tools.registry import register_tool


@register_tool(
    name="current_time",
    description="Get the current date and time. Supports specifying a timezone.",
    parameters={
        "type": "object",
        "properties": {
            "timezone": {
                "type": "string",
                "description": "Timezone like 'Asia/Shanghai', 'UTC'. Default is UTC.",
            }
        },
        "required": [],
    },
)
class CurrentTimeTool(BaseTool):
    @property
    def info(self) -> ToolInfo:
        return self._info  # type: ignore[attr-defined]

    async def execute(self, timezone: str = "UTC") -> ToolResult:
        try:
            if timezone == "UTC":
                tz = UTC
            else:
                from zoneinfo import ZoneInfo

                tz = ZoneInfo(timezone)
            now = datetime.now(tz)
            return ToolResult(content=now.strftime("%Y-%m-%d %H:%M:%S %Z"))
        except Exception as e:
            return ToolResult(content=f"Timezone error: {e}", success=False)

from pathlib import Path
from tools.base import BaseTool, ToolInfo, ToolResult
from tools.registry import register_tool


@register_tool(
    name="read_file",
    description=(
        "Read the content of a file from the filesystem. "
        "Supports text files and can read specific line ranges."
    ),
    parameters={
        "type": "object",
        "properties": {
            "filepath": {
                "type": "string",
                "description": "Absolute or relative path to the file.",
            },
            "start_line": {
                "type": "integer",
                "description": "Start line number (1-indexed). Default is 1.",
            },
            "end_line": {
                "type": "integer",
                "description": "End line number (inclusive). Omit to read until EOF.",
            },
            "max_bytes": {
                "type": "integer",
                "description": "Maximum bytes to read. Default is 102400 (100KB).",
            },
        },
        "required": ["filepath"],
    },
)
class ReadFileTool(BaseTool):
    @property
    def info(self) -> ToolInfo:
        return self._info

    async def execute(
        self,
        filepath: str,
        start_line: int = 1,
        end_line: int | None = None,
        max_bytes: int = 102400,
    ) -> ToolResult:
        try:
            target = Path(filepath).resolve()

            if not target.exists():
                return ToolResult(content=f"File not found: {target}", success=False)

            if not target.is_file():
                return ToolResult(content=f"Not a file: {target}", success=False)

            size = target.stat().st_size
            if size > max_bytes * 10:
                return ToolResult(
                    content=(
                        f"File too large ({size} bytes). "
                        f"Use start_line/end_line to read a specific range, "
                        f"or increase max_bytes (current limit: {max_bytes})."
                    ),
                    success=False,
                )

            text = target.read_text(encoding="utf-8", errors="replace")
            lines = text.splitlines(keepends=True)

            if start_line < 1:
                start_line = 1

            sliced = lines[start_line - 1 : end_line]
            content = "".join(sliced)

            total_lines = len(lines)
            info_parts = [f"File: {target} ({total_lines} lines, {size} bytes)"]
            if start_line != 1 or end_line is not None:
                actual_end = min(end_line or total_lines, total_lines)
                info_parts.append(f"Showing lines {start_line}-{actual_end}")

            return ToolResult(
                content="\n".join(info_parts) + "\n---\n" + content.rstrip()
            )
        except PermissionError:
            return ToolResult(content=f"Permission denied: {target}", success=False)
        except Exception as e:
            return ToolResult(content=f"Read failed: {e}", success=False)

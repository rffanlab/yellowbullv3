from pathlib import Path
from tools.base import BaseTool, ToolInfo, ToolResult
from tools.registry import register_tool


@register_tool(
    name="write_file",
    description=(
        "Write content to a file. Creates parent directories if they don't exist. "
        "Supports append mode and force overwrite control."
    ),
    parameters={
        "type": "object",
        "properties": {
            "filepath": {
                "type": "string",
                "description": "Absolute path to the file. Must start with a drive letter (e.g., D:\\) or root (/). Relative paths are NOT accepted.",
            },
            "content": {
                "type": "string",
                "description": "Content to write to the file.",
            },
            "append": {
                "type": "boolean",
                "description": "If true, append to existing file. Default is false (overwrite).",
            },
            "create_dirs": {
                "type": "boolean",
                "description": "Create parent directories if missing. Default is true.",
            },
        },
        "required": ["filepath", "content"],
    },
)
class WriteFileTool(BaseTool):
    @property
    def info(self) -> ToolInfo:
        return self._info

    async def execute(
        self,
        filepath: str,
        content: str,
        append: bool = False,
        create_dirs: bool = True,
    ) -> ToolResult:
        try:
            if not filepath.startswith(("/", "\\",)) and ":" not in filepath[:2]:
                return ToolResult(
                    content=f"Relative paths are not accepted. Please provide an absolute path (e.g., D:\\project\\file.py). Got: {filepath}",
                    success=False,
                    error="relative_path_not_allowed",
                )

            target = Path(filepath).resolve()

            if target.exists() and target.is_dir():
                return ToolResult(
                    content=f"Path exists and is a directory: {target}", success=False
                )

            if create_dirs:
                target.parent.mkdir(parents=True, exist_ok=True)

            mode = "a" if append else "w"
            before_size = target.stat().st_size if target.exists() else 0

            with open(target, mode, encoding="utf-8") as f:
                f.write(content)

            after_size = target.stat().st_size
            action = "appended to" if append else "wrote to"
            return ToolResult(
                content=(
                    f"Successfully {action} {target}\n"
                    f"Previous size: {before_size} bytes\n"
                    f"New size: {after_size} bytes\n"
                    f"Written: {len(content)} chars"
                )
            )
        except PermissionError:
            return ToolResult(content=f"Permission denied: {target}", success=False)
        except Exception as e:
            return ToolResult(content=f"Write failed: {e}", success=False)

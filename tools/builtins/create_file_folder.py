from pathlib import Path
from tools.base import BaseTool, ToolInfo, ToolResult
from tools.registry import register_tool


@register_tool(
    name="create_file_folder",
    description=(
        "Create a new file or directory. For files, optional initial content can be provided. "
        "For directories, supports recursive creation."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute path to create. Must start with a drive letter (e.g., D:\\) or root (/). Relative paths are NOT accepted.",
            },
            "kind": {
                "type": "string",
                "enum": ["file", "directory"],
                "description": "Whether to create a file or directory. Default is 'file'.",
            },
            "content": {
                "type": "string",
                "description": "Initial content for the file. Ignored if kind='directory'.",
            },
            "recursive": {
                "type": "boolean",
                "description": "Create parent directories if missing. Default is true.",
            },
        },
        "required": ["path", "kind"],
    },
)
class CreateFileFolderTool(BaseTool):
    @property
    def info(self) -> ToolInfo:
        return self._info

    async def execute(
        self,
        path: str,
        kind: str = "file",
        content: str | None = None,
        recursive: bool = True,
    ) -> ToolResult:
        try:
            from pathlib import PureWindowsPath

            if not path.startswith(("/", "\\",)) and ":" not in path[:2]:
                return ToolResult(
                    content=f"Relative paths are not accepted. Please provide an absolute path (e.g., D:\\project\\file.py). Got: {path}",
                    success=False,
                    error="relative_path_not_allowed",
                )

            target = Path(path).resolve()

            if target.exists():
                return ToolResult(
                    content=f"Path already exists: {target}", success=False
                )

            if kind == "directory":
                if recursive:
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.mkdir(exist_ok=True)
                return ToolResult(content=f"Directory created: {target}")

            elif kind == "file":
                if recursive and not target.parent.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)

                with open(target, "w", encoding="utf-8") as f:
                    if content:
                        f.write(content)

                size = target.stat().st_size
                return ToolResult(
                    content=f"File created: {target} ({size} bytes)"
                )

            else:
                return ToolResult(
                    content=f"Invalid kind '{kind}'. Must be 'file' or 'directory'.",
                    success=False,
                )

        except PermissionError:
            return ToolResult(content=f"Permission denied: {target}", success=False)
        except FileExistsError:
            return ToolResult(content=f"Path already exists: {target}", success=False)
        except Exception as e:
            return ToolResult(content=f"Create failed: {e}", success=False)

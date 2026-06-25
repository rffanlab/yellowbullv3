import asyncio
from pathlib import Path
from tools.base import BaseTool, ToolInfo, ToolResult
from tools.registry import register_tool


DEFAULT_TIMEOUT = 60


@register_tool(
    name="execute_command",
    description=(
        "Execute a shell command and return its output. "
        "Supports timeout control and working directory specification."
    ),
    parameters={
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute.",
            },
            "workdir": {
                "type": "string",
                "description": "Working directory for the command. Default is current directory.",
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds. Default is 60.",
            },
        },
        "required": ["command"],
    },
)
class ExecuteCommandTool(BaseTool):
    max_retries: int = 0

    @property
    def info(self) -> ToolInfo:
        return self._info

    async def execute(
        self,
        command: str,
        workdir: str | None = None,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> ToolResult:
        try:
            cwd = Path(workdir).resolve() if workdir else None
            if cwd and not cwd.exists():
                return ToolResult(content=f"Working directory not found: {cwd}", success=False)

            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(cwd) if cwd else None,
            )

            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                proc.kill()
                return ToolResult(
                    content=f"Command timed out after {timeout}s:\n{command}", success=False
                )

            out_text = stdout.decode("utf-8", errors="replace").rstrip()
            err_text = stderr.decode("utf-8", errors="replace").rstrip()

            parts = [f"$ {command}"]
            if cwd:
                parts.append(f"  (in {cwd})")

            if out_text:
                parts.append(out_text)
            if err_text and proc.returncode != 0:
                parts.append(f"STDERR:\n{err_text}")

            status = "succeeded" if proc.returncode == 0 else f"failed (exit {proc.returncode})"
            parts.insert(0, f"[{status}]")

            return ToolResult(
                content="\n".join(parts), success=proc.returncode == 0
            )
        except Exception as e:
            return ToolResult(content=f"Execution failed: {e}", success=False)

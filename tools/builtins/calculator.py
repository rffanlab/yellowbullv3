"""Calculator tool for math expressions."""

import math

from tools.base import BaseTool, ToolInfo, ToolResult
from tools.registry import register_tool


@register_tool(
    name="calculator",
    description=(
        "Evaluate a mathematical expression. "
        "Supports arithmetic, powers, trig functions, etc."
    ),
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Math expression like '2 + 3 * 4', 'sin(3.14)', 'sqrt(16)'",
            }
        },
        "required": ["expression"],
    },
)
class CalculatorTool(BaseTool):
    @property
    def info(self) -> ToolInfo:
        return self._info  # type: ignore[attr-defined]

    async def execute(self, expression: str) -> ToolResult:
        try:
            # Safe math-only evaluation
            allowed_names = {
                "abs": abs,
                "round": round,
                "min": min,
                "max": max,
                "sum": sum,
                "pow": pow,
                "sqrt": math.sqrt,
                "sin": math.sin,
                "cos": math.cos,
                "tan": math.tan,
                "log": math.log,
                "pi": math.pi,
                "e": math.e,
            }
            result = eval(expression, {"__builtins__": {}}, allowed_names)  # noqa: S307
            return ToolResult(content=str(result))
        except Exception as e:
            return ToolResult(content=f"Calculation error: {e}", success=False)

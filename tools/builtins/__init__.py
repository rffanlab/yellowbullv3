"""Import all built-in tools to trigger registration."""

from tools.builtins.calculator import CalculatorTool  # noqa: F401
from tools.builtins.command_execute import ExecuteCommandTool  # noqa: F401
from tools.builtins.create_file_folder import CreateFileFolderTool  # noqa: F401
from tools.builtins.current_time import CurrentTimeTool  # noqa: F401
from tools.builtins.file_read import ReadFileTool  # noqa: F401
from tools.builtins.file_write import WriteFileTool  # noqa: F401
from tools.builtins.web_search import WebSearchTool  # noqa: F401

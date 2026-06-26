"""Interactive REPL for YellowBull Agent with rich terminal rendering."""

import os
from typing import Any

from config.settings import load_settings
from core.agent import Agent, ChatRequest
from llm.factory import create_llm
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

# Import built-in tools to trigger registration
import tools.builtins  # noqa: F401


console = Console()


def _print_welcome(work_dir: str | None = None) -> None:
    """Print welcome banner."""
    lines = ["[bold blue]YellowBull Agent[/bold blue]\n"]
    if work_dir:
        lines.append(f"[dim]工作目录: {work_dir}[/dim]\n")
    lines.append("[dim]交互式命令行 · 输入 /help 查看命令[/dim]")
    console.print(Panel.fit(
        "\n".join(lines),
        border_style="blue",
    ))


def _ask_work_dir(agent: Agent, user_id: str) -> tuple[str, str]:
    """Ask user for working directory and set it on the session.

    Returns (session_id, work_dir).
    """
    while True:
        try:
            work_dir = console.input("[bold yellow]请输入工作目录路径（文件操作将在此目录下进行）:[/bold yellow] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]再见 👋[/dim]")
            raise SystemExit

        if not work_dir:
            console.print("[red]工作目录不能为空，请重新输入。[/red]")
            continue

        # Normalize path
        work_dir = os.path.abspath(work_dir)

        # Verify directory exists, create if not
        if not os.path.isdir(work_dir):
            try:
                os.makedirs(work_dir, exist_ok=True)
                console.print(f"[green]✓ 已创建工作目录: {work_dir}[/green]")
            except OSError as e:
                console.print(f"[red]无法创建目录 {work_dir}: {e}[/red]")
                continue

        # Create session and set work_dir
        session = agent.session_manager.create(user_id)
        session.work_dir = work_dir
        return session.session_id, work_dir


def _print_help() -> None:
    """Print available commands."""
    console.print(Panel(
        "[bold]可用命令:[/bold]\n\n"
        "  [cyan]/help[/cyan]      显示帮助信息\n"
        "  [cyan]/new[/cyan]       创建新会话（继承工作目录）\n"
        "  [cyan]/workdir[/cyan]    查看或更改工作目录\n"
        "  [cyan]/sessions[/cyan]   列出所有会话\n"
        "  [cyan]/history[/cyan]    查看当前会话历史\n"
        "  [cyan]/clear[/cyan]      清屏\n"
        "  [cyan]/quit[/cyan]       退出程序",
        title="[bold]帮助[/bold]",
        border_style="dim",
    ))


def _print_tool_call(tool_name: str, args: dict[str, Any], result_content: str) -> None:
    """Render a tool call with its arguments and result."""
    args_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
    console.print(Panel(
        f"[bold]参数:[/bold] {args_str}\n\n[dim]{result_content[:500]}[/dim]",
        title=f"[yellow]🔧 工具调用: {tool_name}[/yellow]",
        border_style="yellow",
    ))


def _print_response(content: str) -> None:
    """Render assistant response as Markdown."""
    console.print(Markdown(content))


def _print_sessions(sessions: list[dict]) -> None:
    """Print session list in a table."""
    from rich.table import Table

    if not sessions:
        console.print("[dim]暂无会话[/dim]")
        return

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Session ID", style="dim", max_width=36)
    table.add_column("User", style="green")
    table.add_column("Messages", justify="right")
    table.add_column("Updated", style="dim")

    for s in sessions[:20]:
        table.add_row(
            s.get("session_id", "")[:36],
            s.get("user_id", ""),
            str(s.get("message_count", 0)),
            s.get("updated_at", "")[:19],
        )

    console.print(table)


def _print_history(messages: list[dict]) -> None:
    """Print message history for current session."""
    if not messages:
        console.print("[dim]当前会话暂无消息[/dim]")
        return

    for msg in messages[-20:]:
        role = msg.get("role", "?")
        content = (msg.get("content", "") or "")[:200]
        style = {"user": "green", "assistant": "blue", "tool": "yellow"}.get(role, "white")
        console.print(f"[{style}]{role.upper()}:[/{style}] {content}")


async def run_cli(config_path: str | None = None) -> None:
    """Start the interactive CLI REPL."""
    settings = load_settings(config_path)

    # Create LLM from active provider config
    active_provider = settings.llm.active
    provider_config = settings.llm.providers.get(active_provider, {})
    if hasattr(provider_config, "model_dump"):
        provider_config = provider_config.model_dump()
    llm = create_llm(active_provider, provider_config)
    agent = Agent(settings, llm)

    user_id = "cli_user"

    # Ask for working directory on startup
    current_session_id, work_dir = _ask_work_dir(agent, user_id)
    _print_welcome(work_dir)

    while True:
        try:
            input_text = console.input("\n[bold green]You[/bold green]: ")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]再见 👋[/dim]")
            break

        text = input_text.strip()
        if not text:
            continue

        # ── Command handling ────────────────────────────────────────
        if text.startswith("/"):
            cmd = text.split()[0].lower()

            if cmd in ("/quit", "/exit", "/q"):
                console.print("[dim]再见 👋[/dim]")
                break

            elif cmd == "/help":
                _print_help()

            elif cmd == "/workdir":
                if work_dir:
                    console.print(f"[dim]当前工作目录: {work_dir}[/dim]")
                else:
                    console.print("[dim]未设置工作目录[/dim]")
                # Allow changing with /workdir <path>
                parts = text.split(maxsplit=1)
                if len(parts) > 1:
                    new_dir = os.path.abspath(parts[1].strip())
                    if not os.path.isdir(new_dir):
                        try:
                            os.makedirs(new_dir, exist_ok=True)
                        except OSError as e:
                            console.print(f"[red]无法创建目录 {new_dir}: {e}[/red]")
                            continue
                    # Update current session work_dir
                    if current_session_id:
                        sess = agent.session_manager.get(current_session_id)
                        if sess:
                            sess.work_dir = new_dir
                    work_dir = new_dir
                    console.print(f"[green]✓ 工作目录已更改为: {work_dir}[/green]")

            elif cmd == "/new":
                # Create new session inheriting work_dir from previous
                old_work_dir = work_dir
                if current_session_id:
                    sess = agent.session_manager.get(current_session_id)
                    if sess and sess.work_dir:
                        old_work_dir = sess.work_dir
                new_session = agent.session_manager.create(user_id)
                new_session.work_dir = old_work_dir
                current_session_id = new_session.session_id
                work_dir = old_work_dir
                console.print("[green]✓ 已创建新会话[/green]")

            elif cmd == "/sessions":
                raw = agent.session_manager.list_sessions(user_id=user_id)
                session_dicts = []
                for sid, sess in raw:
                    d = sess.model_dump()
                    d["session_id"] = sid
                    session_dicts.append(d)
                _print_sessions(session_dicts)

            elif cmd == "/history":
                if current_session_id:
                    session = agent.session_manager.get(current_session_id)
                    if session and session.messages:
                        msgs = [{"role": m.role, "content": m.content} for m in session.messages]
                        _print_history(msgs)
                    else:
                        console.print("[dim]当前会话暂无消息[/dim]")
                else:
                    console.print("[dim]请先发送一条消息创建会话[/dim]")

            elif cmd == "/clear":
                console.clear()

            else:
                console.print(f"[red]未知命令: {cmd}，输入 /help 查看帮助[/red]")

            continue

        # ── Chat request ────────────────────────────────────────────
        console.print()  # blank line before response

        try:
            response = await agent.chat(
                ChatRequest(
                    session_id=current_session_id,
                    user_id=user_id,
                    message=text,
                )
            )
            current_session_id = response.session_id

            # Show tool results if any
            for tr in response.tool_results:
                console.print()
                _print_tool_call(tr.get("tool", "unknown"), {}, tr["content"])

            # Show final answer
            console.print()
            _print_response(response.content or "(无回复)")

        except Exception as e:
            console.print(f"[red]请求失败: {e}[/red]")

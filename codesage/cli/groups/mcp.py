"""MCP command group for CodeSage CLI.

Provides commands for running and configuring the MCP server
for AI IDE integration.
"""

import asyncio
import json
from pathlib import Path
from typing import Optional

import typer
from rich.panel import Panel
from rich.syntax import Syntax

from codesage.cli.utils.console import (
    get_console,
    get_stderr_console,
    print_error,
    print_success,
    set_mcp_stdio_mode,
)
from codesage.cli.utils.decorators import handle_errors

app = typer.Typer(help="MCP server for AI IDE integration")


@app.command("serve")
@handle_errors
def serve(
    path: str = typer.Argument(".", help="Project directory (ignored if --global is used)"),
    transport: str = typer.Option(
        "stdio",
        "--transport",
        "-t",
        help="Transport type: stdio (single client) or sse (HTTP, multi-client)",
    ),
    host: str = typer.Option(
        "localhost",
        "--host",
        help="Host for SSE transport (default: localhost)",
    ),
    port: int = typer.Option(
        8080,
        "--port",
        "-p",
        help="Port for SSE transport (default: 8080)",
    ),
    global_mode: bool = typer.Option(
        False,
        "--global",
        "-g",
        help="Run in global mode (serves all indexed projects)",
    ),
) -> None:
    """Start the MCP server.

    Two modes:

    1. Project mode (default): Serves a specific project

       codesage mcp serve /path/to/project

    2. Global mode: Serves all indexed projects

       codesage mcp serve --global

    Transports:

    - stdio: Single client, process-based (default)

    - sse: Multiple clients, HTTP-based

    Examples:

      codesage mcp serve

      codesage mcp serve --global

      codesage mcp serve --global -t sse -p 8080
    """
    console = get_console()
    stderr_console = get_stderr_console()

    # Validate transport
    if transport not in ["stdio", "sse"]:
        print_error(f"Invalid transport: {transport}. Must be 'stdio' or 'sse'.")
        raise typer.Exit(1)

    # Enable stdio mode to redirect all console output to stderr
    # This prevents corrupting the MCP JSON-RPC protocol on stdout
    if transport == "stdio":
        set_mcp_stdio_mode(True)

    if global_mode:
        # Global mode: serve all projects
        from codesage.mcp.global_server import GlobalCodeSageMCPServer

        if transport == "sse":
            console.print(
                Panel(
                    f"[bold]Mode:[/bold] Global (all projects)\n"
                    f"[bold]Transport:[/bold] {transport}\n"
                    f"[bold]Endpoint:[/bold] http://{host}:{port}/sse",
                    title="CodeSage Global MCP Server",
                    border_style="green",
                )
            )

        try:
            global_server = GlobalCodeSageMCPServer()

            if transport == "stdio":
                asyncio.run(global_server.run_stdio())
            else:
                asyncio.run(global_server.run_sse(host=host, port=port))

        except KeyboardInterrupt:
            if transport == "sse":
                console.print("\n[yellow]Server stopped by user[/yellow]")
        except Exception as e:
            print_error(f"Server error: {e}")
            raise typer.Exit(1)
    else:
        # Project mode: serve specific project
        from codesage.mcp.server import CodeSageMCPServer
        from codesage.utils.config import Config

        project_path = Path(path).resolve()

        try:
            config = Config.load(project_path)
        except FileNotFoundError:
            print_error(f"Project not initialized at {project_path}")
            stderr_console.print("Run 'codesage init' first")
            raise typer.Exit(1)

        if transport == "sse":
            console.print(
                Panel(
                    f"[bold]Mode:[/bold] Project\n"
                    f"[bold]Project:[/bold] {config.project_name}\n"
                    f"[bold]Path:[/bold] {project_path}\n"
                    f"[bold]Transport:[/bold] {transport}\n"
                    f"[bold]Endpoint:[/bold] http://{host}:{port}/sse",
                    title="CodeSage MCP Server",
                    border_style="cyan",
                )
            )

        try:
            server = CodeSageMCPServer(project_path)

            if transport == "stdio":
                asyncio.run(server.run_stdio())
            else:
                asyncio.run(server.run_sse(host=host, port=port))

        except KeyboardInterrupt:
            if transport == "sse":
                console.print("\n[yellow]Server stopped by user[/yellow]")
        except Exception as e:
            print_error(f"Server error: {e}")
            raise typer.Exit(1)


@app.command("setup")
@handle_errors
def setup(
    transport: str = typer.Option(
        "stdio",
        "--transport",
        "-t",
        help="Transport type: stdio or sse",
    ),
    port: int = typer.Option(
        8080,
        "--port",
        "-p",
        help="Port for SSE transport",
    ),
) -> None:
    """Show MCP server configuration for your AI IDE.

    Displays the JSON configuration to add CodeSage as an
    MCP server in any AI IDE that supports the MCP protocol.

    Examples:

      codesage mcp setup

      codesage mcp setup -t sse -p 8080
    """
    console = get_console()

    # Always use bare "codesage" for portable cross-platform config.
    # Users should have it on PATH via pip/pipx install.
    codesage_cmd = "codesage"

    console.print(
        Panel(
            f"[bold]Server:[/bold] codesage\n"
            f"[bold]Mode:[/bold] Global (all indexed projects)\n"
            f"[bold]Transport:[/bold] {transport}",
            title="CodeSage MCP Server Setup",
            border_style="green",
        )
    )
    console.print()

    if transport == "stdio":
        mcp_config = {
            "mcpServers": {
                "codesage": {
                    "command": codesage_cmd,
                    "args": ["mcp", "serve", "--global"]
                }
            }
        }

        console.print("[bold]Add this to your AI IDE's MCP configuration:[/bold]\n")
        console.print(Syntax(json.dumps(mcp_config, indent=2), "json", theme="monokai"))
        console.print()
        console.print("[bold]Available tools (12):[/bold]")
        console.print("  [dim]Global:[/dim]  list_projects, get_developer_profile")
        console.print("  [dim]Search:[/dim]  search_code, get_file_context, get_stats")
        console.print("  [dim]Analysis:[/dim] review_code, analyze_security, explain_concept")
        console.print("  [dim]Context:[/dim]  suggest_approach, trace_flow, find_examples, recommend_pattern")
        console.print()
        console.print("[dim]Verify with: codesage mcp test[/dim]")

    else:
        console.print(f"[bold]1. Start the server:[/bold]")
        console.print(f"   [cyan]codesage mcp serve --global -t sse -p {port}[/cyan]\n")

        endpoint = f"http://localhost:{port}/sse"

        console.print(f"[bold]2. Configure your AI IDE with this endpoint:[/bold]")
        console.print(f"   [cyan]{endpoint}[/cyan]\n")

        sse_config = {
            "mcpServers": {
                "codesage": {
                    "url": endpoint
                }
            }
        }
        console.print(Syntax(json.dumps(sse_config, indent=2), "json", theme="monokai"))


@app.command("test")
@handle_errors
def test(
    path: str = typer.Argument(".", help="Project directory"),
) -> None:
    """Test MCP server functionality.

    Verifies that MCP tools work correctly against an
    indexed project.

    Examples:

      codesage mcp test

      codesage mcp test /path/to/project
    """
    from codesage.mcp import check_mcp_available

    check_mcp_available()

    from codesage.mcp.server import CodeSageMCPServer

    console = get_console()
    project_path = Path(path).resolve()

    console.print("[cyan]Testing MCP server...[/cyan]\n")

    # Initialize server
    try:
        server = CodeSageMCPServer(project_path)
        print_success("Server initialized")
    except Exception as e:
        print_error(f"Server initialization failed: {e}")
        raise typer.Exit(1)

    # Test each tool
    async def run_tests():
        results = []

        # Test search_code
        try:
            result = await server._tool_search_code({"query": "main entry point", "limit": 3})
            items = result.get("results", [])
            count = len(items)
            detail = ""
            if items:
                top = items[0]
                name = top.get("name", "?")
                file = top.get("file", "?")
                sim = top.get("similarity", 0)
                detail = f" → top: {name} in {file} ({sim:.0%})"
            results.append(("search_code", count > 0, f"{count} results{detail}"))
        except Exception as e:
            results.append(("search_code", False, str(e)))

        # Test get_stats
        try:
            result = await server._tool_get_stats({"detailed": False})
            inner = result.get("results", {})
            files = inner.get("files_indexed", 0)
            elements = inner.get("code_elements", 0)
            lang = inner.get("language", "?")
            results.append(("get_stats", True, f"{files} files, {elements} elements ({lang})"))
        except Exception as e:
            results.append(("get_stats", False, str(e)))

        # Test get_file_context
        try:
            result = await server._tool_get_file_context({"file_path": "README.md"})
            inner = result.get("results", {})
            lines = inner.get("line_count", 0)
            defs = len(inner.get("definitions", []))
            results.append(("get_file_context", True, f"README.md: {lines} lines, {defs} definitions"))
        except Exception as e:
            results.append(("get_file_context", False, str(e)))

        # Test explain_concept
        try:
            result = await server._tool_explain_concept({"concept": "main entry point", "depth": "quick"})
            items = result.get("results", [])
            narrative = result.get("narrative", "")
            preview = narrative[:80] + "..." if len(narrative) > 80 else narrative
            results.append(("explain_concept", len(items) > 0, f"{len(items)} results, narrative: {preview}"))
        except Exception as e:
            results.append(("explain_concept", False, str(e)))

        # Test trace_flow
        try:
            result = await server._tool_trace_flow({"element_name": "main", "direction": "both"})
            inner = result.get("results", {})
            callers = inner.get("callers", [])
            callees = inner.get("callees", [])
            detail_parts = []
            if callers:
                detail_parts.append(f"callers: {', '.join(c.get('name', '?') for c in callers[:3])}")
            if callees:
                detail_parts.append(f"callees: {', '.join(c.get('name', '?') for c in callees[:3])}")
            detail = " → " + "; ".join(detail_parts) if detail_parts else ""
            results.append(("trace_flow", True, f"{len(callers)} callers, {len(callees)} callees{detail}"))
        except Exception as e:
            results.append(("trace_flow", False, str(e)))

        # Test analyze_security
        try:
            result = await server._tool_analyze_security({"path": ".", "severity": "high"})
            inner = result.get("results", {})
            findings = inner.get("findings", [])
            total = inner.get("total_findings", len(findings))
            results.append(("analyze_security", True, f"{total} findings"))
        except Exception as e:
            results.append(("analyze_security", False, str(e)))

        # Test suggest_approach
        try:
            result = await server._tool_suggest_approach({"task": "add caching to embeddings"})
            inner = result.get("results", {})
            code = inner.get("relevant_code", [])
            patterns = inner.get("patterns", [])
            results.append(("suggest_approach", True, f"{len(code)} relevant code, {len(patterns)} patterns"))
        except Exception as e:
            results.append(("suggest_approach", False, str(e)))

        # Test find_examples
        try:
            result = await server._tool_find_examples({"pattern": "error handling", "limit": 3})
            items = result.get("results", [])
            results.append(("find_examples", True, f"{len(items)} examples"))
        except Exception as e:
            results.append(("find_examples", False, str(e)))

        # Test review_code
        try:
            result = await server._tool_review_code({"scope": "staged"})
            inner = result.get("results", {})
            summary = inner.get("summary", "?")
            issues = len(inner.get("issues", []))
            results.append(("review_code", True, f"{issues} issues | {summary[:60]}"))
        except Exception as e:
            results.append(("review_code", False, str(e)))

        # Test recommend_pattern
        try:
            result = await server._tool_recommend_pattern({"context": "error handling", "limit": 3})
            items = result.get("results", [])
            names = [p.get("name", "?") for p in items[:3]]
            results.append(("recommend_pattern", True, f"{len(items)} patterns: {', '.join(names)}"))
        except Exception as e:
            results.append(("recommend_pattern", False, str(e)))

        return results

    results = asyncio.run(run_tests())

    # Display results
    console.print("\n[bold]Tool Test Results:[/bold]\n")

    passed = 0
    for tool, success, message in results:
        if success:
            console.print(f"  [green]✓[/green] {tool}: {message}")
            passed += 1
        else:
            console.print(f"  [red]✗[/red] {tool}: {message}")

    console.print()

    total = len(results)
    if passed == total:
        print_success(f"All {total} tools verified!")
    else:
        console.print(f"[yellow]{passed}/{total} tools passed[/yellow]")

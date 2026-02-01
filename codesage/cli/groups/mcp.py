"""MCP command group for CodeSage CLI.

Provides commands for running and managing the MCP server
for Claude Desktop integration.
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Optional

import typer
from rich.panel import Panel
from rich.syntax import Syntax

from codesage.cli.utils.console import get_console, print_error, print_success
from codesage.cli.utils.decorators import handle_errors

app = typer.Typer(help="MCP server for Claude Desktop integration")


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
    """Start the MCP server with stdio or HTTP/SSE transport.

    **Two modes:**

    1. Project mode (default): Serves a specific project
       codesage mcp serve /path/to/project

    2. Global mode: Serves all indexed projects with project_name parameter
       codesage mcp serve --global

    **Transports:**
    - stdio: Single client, process-based (Claude Desktop)
    - sse: Multiple clients, HTTP-based (web apps, remote)

    Example usage:
      - Project mode: codesage mcp serve
      - Global mode: codesage mcp serve --global
      - HTTP mode: codesage mcp serve --global -t sse -p 8080
    """
    console = get_console()

    # Validate transport
    if transport not in ["stdio", "sse"]:
        print_error(f"Invalid transport: {transport}. Must be 'stdio' or 'sse'.")
        raise typer.Exit(1)

    if global_mode:
        # Global mode: serve all projects
        from codesage.mcp.global_server import GlobalCodeSageMCPServer

        # Only show panel for SSE transport (stdio must have clean stdout for JSON-RPC)
        if transport == "sse":
            console.print(
                Panel(
                    f"[bold]Mode:[/bold] Global (all projects)\n"
                    f"[bold]Transport:[/bold] {transport}\n"
                    f"[bold]Endpoint:[/bold] http://{host}:{port}/sse",
                    title="🌐 CodeSage Global MCP Server",
                    border_style="green",
                )
            )

        try:
            global_server = GlobalCodeSageMCPServer()

            if transport == "stdio":
                asyncio.run(global_server.run_stdio())
            else:  # sse
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

        # Verify project exists
        try:
            config = Config.load(project_path)
        except FileNotFoundError:
            print_error(f"Project not initialized at {project_path}")
            console.print("Run 'codesage init' first")
            raise typer.Exit(1)

        # Only show panel for SSE transport (stdio must have clean stdout for JSON-RPC)
        if transport == "sse":
            console.print(
                Panel(
                    f"[bold]Mode:[/bold] Project\n"
                    f"[bold]Project:[/bold] {config.project_name}\n"
                    f"[bold]Path:[/bold] {project_path}\n"
                    f"[bold]Transport:[/bold] {transport}\n"
                    f"[bold]Endpoint:[/bold] http://{host}:{port}/sse",
                    title="📦 CodeSage MCP Server",
                    border_style="cyan",
                )
            )

    # Run the server
    try:
        asyncio.run(run_mcp_server(project_path, transport=transport, host=host, port=port))
    except KeyboardInterrupt:
        if transport == "sse":
            console.print("\n[dim]Server stopped[/dim]")
        pass


@app.command("install")
@handle_errors
def install(
    path: str = typer.Argument(".", help="Project directory"),
    name: Optional[str] = typer.Option(
        None,
        "--name",
        "-n",
        help="Server name (default: project name)",
    ),
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
    """Show MCP server configuration for different clients.

    Displays configuration examples for:
    - Claude Desktop
    - Cursor
    - Windsurf
    - Custom MCP clients

    Supports both stdio and SSE transports.
    """
    from codesage.utils.config import Config

    console = get_console()
    project_path = Path(path).resolve()

    # Load project config
    try:
        config = Config.load(project_path)
    except FileNotFoundError:
        print_error(f"Project not initialized at {project_path}")
        raise typer.Exit(1)

    if name:
        server_name = name
    elif config.project_name == "codesage":
        server_name = "codesage"
    else:
        server_name = f"codesage-{config.project_name}"

    # Show project info
    console.print(
        Panel(
            f"[bold]Project:[/bold] {config.project_name}\n"
            f"[bold]Path:[/bold] {project_path}\n"
            f"[bold]Server Name:[/bold] {server_name}\n"
            f"[bold]Transport:[/bold] {transport}",
            title="CodeSage MCP Server Configuration",
            border_style="cyan",
        )
    )

    console.print()

    # Multi-project recommendation
    console.print("[bold yellow]� Working with Multiple Projects:[/bold yellow]")
    console.print()
    console.print("  [bold green]Recommended:[/bold green] Use global mode for multi-project access:")
    console.print(f"  [cyan]codesage mcp serve --global[/cyan]")
    console.print()
    console.print("  This serves ALL indexed projects with a single server!")
    console.print("  Tools accept a [bold]project_name[/bold] parameter to target specific projects.")
    console.print()
    console.print("  [dim]Alternative: Run separate servers per project (not recommended)[/dim]")
    console.print()

    # Stdio configuration
    if transport == "stdio":
        console.print("[bold cyan]═══ Stdio Transport (Process-based) ═══[/bold cyan]\n")

        # Claude Desktop
        console.print("[bold]1. Claude Desktop[/bold]")
        console.print("[dim]Config file:[/dim] ~/.config/claude/claude_desktop_config.json (Linux)")
        console.print("[dim]             [/dim] ~/Library/Application Support/Claude/claude_desktop_config.json (macOS)")
        console.print()

        claude_config = {
            "mcpServers": {
                server_name: {
                    "command": "codesage",
                    "args": ["mcp", "serve", str(project_path)]  # Absolute path required
                }
            }
        }
        console.print(Syntax(json.dumps(claude_config, indent=2), "json", theme="monokai"))
        console.print()

        # Cursor
        console.print("[bold]2. Cursor IDE[/bold]")
        console.print("[dim]Settings → Features → MCP Servers[/dim]")
        console.print()

        cursor_config = {
            "mcpServers": {
                server_name: {
                    "command": "codesage",
                    "args": ["mcp", "serve", str(project_path)]  # Absolute path required
                }
            }
        }
        console.print(Syntax(json.dumps(cursor_config, indent=2), "json", theme="monokai"))
        console.print()

        # Windsurf
        console.print("[bold]3. Windsurf[/bold]")
        console.print("[dim]Settings → MCP → Add Server[/dim]")
        console.print()

        windsurf_config = {
            "name": server_name,
            "command": "codesage",
            "args": ["mcp", "serve", str(project_path)]  # Absolute path required
        }
        console.print(Syntax(json.dumps(windsurf_config, indent=2), "json", theme="monokai"))
        console.print()

        # Generic stdio
        console.print("[bold]4. Custom MCP Client (Python)[/bold]")
        console.print()

        python_example = f'''from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def connect():
    params = StdioServerParameters(
        command="codesage",
        args=["mcp", "serve", "/path/to/your/project"] # Replace with your project path
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            print(f"Available tools: {{[t.name for t in tools]}}")
'''
        console.print(Syntax(python_example, "python", theme="monokai"))

    # SSE configuration
    else:
        console.print("[bold cyan]═══ SSE Transport (HTTP-based) ═══[/bold cyan]\n")

        console.print(f"[bold]Start the server first:[/bold]")
        console.print(f"[cyan]codesage mcp serve -t sse -p {port}[/cyan]\n")

        endpoint = f"http://localhost:{port}/sse"

        # Web browser
        console.print("[bold]1. Web Browser / JavaScript[/bold]")
        console.print()

        js_example = f'''import {{ Client }} from "@modelcontextprotocol/sdk/client/index.js";
import {{ SSEClientTransport }} from "@modelcontextprotocol/sdk/client/sse.js";

const transport = new SSEClientTransport(
  new URL("{endpoint}")
);

const client = new Client({{
  name: "{server_name}",
  version: "1.0.0"
}}, {{ capabilities: {{}} }});

await client.connect(transport);

// List tools
const tools = await client.listTools();
console.log("Tools:", tools.tools);

// Search code
const result = await client.callTool({{
  name: "search_code",
  arguments: {{ query: "authentication" }}
}});
'''
        console.print(Syntax(js_example, "javascript", theme="monokai"))
        console.print()

        # Python SSE client
        console.print("[bold]2. Python MCP Client[/bold]")
        console.print()

        python_sse = f'''from mcp import ClientSession
from mcp.client.sse import sse_client

async def connect():
    async with sse_client(url="{endpoint}") as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Search code
            result = await session.call_tool(
                "search_code",
                {{"query": "database query", "limit": 5}}
            )
            print(result)
'''
        console.print(Syntax(python_sse, "python", theme="monokai"))
        console.print()

        # curl example
        console.print("[bold]3. curl / REST API[/bold]")
        console.print()

        curl_example = f'''# Connect to SSE endpoint
curl -N {endpoint}

# The endpoint supports GET requests and streams events
'''
        console.print(Syntax(curl_example, "bash", theme="monokai"))

    # Installation instructions
    console.print()
    console.print("[bold yellow]📝 Next Steps:[/bold yellow]")
    console.print()

    if transport == "stdio":
        console.print("  1. Copy the relevant configuration above")
        console.print("  2. Add it to your MCP client's config file")
        console.print("  3. Restart the client application")
        console.print("  4. CodeSage tools will be available!")
    else:
        console.print(f"  1. Start the server: [cyan]codesage mcp serve -t sse -p {port}[/cyan]")
        console.print(f"  2. Connect clients to: [cyan]{endpoint}[/cyan]")
        console.print("  3. Multiple clients can connect simultaneously")
        console.print("  4. Stop server with Ctrl+C")

    console.print()
    console.print("[dim]💡 Tip: Use `codesage mcp test` to verify the server works[/dim]")


@app.command("uninstall")
@handle_errors
def uninstall(
    name: str = typer.Argument(..., help="Server name to remove"),
    client: str = typer.Option(
        "claude",
        "--client",
        "-c",
        help="Client type: claude, cursor, windsurf",
    ),
) -> None:
    """Show instructions to remove CodeSage MCP server from a client."""
    import sys

    console = get_console()

    console.print(
        Panel(
            f"[bold]Removing Server:[/bold] {name}\n"
            f"[bold]Client:[/bold] {client}",
            title="MCP Server Removal",
            border_style="yellow",
        )
    )
    console.print()

    if client == "claude":
        # Find Claude Desktop config
        if sys.platform == "darwin":
            config_file = Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
        elif sys.platform == "win32":
            config_file = Path(os.environ.get("APPDATA", "")) / "Claude" / "claude_desktop_config.json"
        else:
            config_file = Path.home() / ".config" / "claude" / "claude_desktop_config.json"

        console.print(f"[bold]Config file:[/bold] {config_file}\n")

        if config_file.exists():
            console.print("[yellow]To remove the server:[/yellow]")
            console.print(f'  1. Open {config_file}')
            console.print(f'  2. Remove the "{name}" entry from "mcpServers"')
            console.print("  3. Save the file")
            console.print("  4. Restart Claude Desktop")
        else:
            console.print("[dim]Config file not found. Server may not be installed.[/dim]")

    elif client == "cursor":
        console.print("[bold]To remove from Cursor:[/bold]")
        console.print("  1. Open Cursor Settings")
        console.print("  2. Go to Features → MCP Servers")
        console.print(f'  3. Find and remove "{name}"')
        console.print("  4. Restart Cursor")

    elif client == "windsurf":
        console.print("[bold]To remove from Windsurf:[/bold]")
        console.print("  1. Open Windsurf Settings")
        console.print("  2. Go to MCP")
        console.print(f'  3. Find and remove "{name}"')
        console.print("  4. Restart Windsurf")

    else:
        print_error(f"Unknown client: {client}")
        console.print("  Supported clients: claude, cursor, windsurf")


@app.command("list")
@handle_errors
def list_servers(
    client: str = typer.Option(
        "claude",
        "--client",
        "-c",
        help="Client type: claude, cursor, windsurf, all",
    ),
) -> None:
    """Show where to find MCP server configurations."""
    import sys

    console = get_console()

    console.print(
        Panel(
            "[bold]MCP Server Configuration Locations[/bold]",
            border_style="cyan",
        )
    )
    console.print()

    if client in ("claude", "all"):
        console.print("[bold cyan]Claude Desktop[/bold cyan]")

        if sys.platform == "darwin":
            config_file = Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
        elif sys.platform == "win32":
            config_file = Path(os.environ.get("APPDATA", "")) / "Claude" / "claude_desktop_config.json"
        else:
            config_file = Path.home() / ".config" / "claude" / "claude_desktop_config.json"

        console.print(f"  📁 Config: {config_file}")

        if config_file.exists():
            try:
                with open(config_file) as f:
                    claude_config = json.load(f)
                    servers = claude_config.get("mcpServers", {})

                    if servers:
                        console.print(f"  ✅ Found {len(servers)} server(s):")
                        for name in servers.keys():
                            is_codesage = "codesage" in str(servers[name])
                            icon = "  �" if is_codesage else "  🔧"
                            console.print(f"{icon} {name}")
                    else:
                        console.print("  [dim]No servers configured[/dim]")
            except Exception as e:
                console.print(f"  [yellow]Could not read config: {e}[/yellow]")
        else:
            console.print("  [dim]Config file not found[/dim]")

        console.print()

    if client in ("cursor", "all"):
        console.print("[bold cyan]Cursor IDE[/bold cyan]")
        console.print("  📁 Settings → Features → MCP Servers")
        console.print("  [dim]Check Cursor settings UI for configured servers[/dim]")
        console.print()

    if client in ("windsurf", "all"):
        console.print("[bold cyan]Windsurf[/bold cyan]")
        console.print("  📁 Settings → MCP")
        console.print("  [dim]Check Windsurf settings UI for configured servers[/dim]")
        console.print()

    if client == "all":
        console.print("[dim]💡 Tip: Use --client to filter by specific client[/dim]")


@app.command("config")
@handle_errors
def show_config(
    path: str = typer.Argument(".", help="Project directory"),
) -> None:
    """Show MCP server configuration for a project.

    Displays the configuration that would be added to Claude Desktop.
    """
    from codesage.utils.config import Config

    console = get_console()
    project_path = Path(path).resolve()

    try:
        config = Config.load(project_path)
    except FileNotFoundError:
        print_error(f"Project not initialized at {project_path}")
        raise typer.Exit(1)

    if config.project_name == "codesage":
        server_name = "codesage"
    else:
        server_name = f"codesage-{config.project_name}"

    mcp_config = {
        server_name: {
            "command": "codesage",
            "args": ["mcp", "serve", str(project_path)],
        }
    }

    console.print(
        Panel(
            f"[bold]Project:[/bold] {config.project_name}\n"
            f"[bold]Path:[/bold] {project_path}",
            title="MCP Server Configuration",
            border_style="blue",
        )
    )

    console.print("\n[bold]Add this to your Claude Desktop config:[/bold]\n")
    console.print(Syntax(json.dumps(mcp_config, indent=2), "json", theme="monokai"))

    console.print("\n[dim]Or run: codesage mcp install[/dim]")


@app.command("test")
@handle_errors
def test(
    path: str = typer.Argument(".", help="Project directory"),
) -> None:
    """Test MCP server functionality.

    Verifies that all MCP tools work correctly.
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
            result = await server._tool_search_code({"query": "function", "limit": 3})
            results.append(("search_code", True, f"{result.get('count', 0)} results"))
        except Exception as e:
            results.append(("search_code", False, str(e)))

        # Test get_stats
        try:
            result = await server._tool_get_stats({"detailed": False})
            results.append(("get_stats", True, f"{result.get('code_elements', 0)} elements"))
        except Exception as e:
            results.append(("get_stats", False, str(e)))

        # Test get_file_context (with a common file)
        try:
            result = await server._tool_get_file_context({"file_path": "README.md"})
            has_content = "content" in result or "error" in result
            results.append(("get_file_context", has_content, "OK" if has_content else "No result"))
        except Exception as e:
            results.append(("get_file_context", False, str(e)))

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

    if passed == len(results):
        print_success(f"All {len(results)} tests passed!")
    else:
        console.print(f"[yellow]{passed}/{len(results)} tests passed[/yellow]")

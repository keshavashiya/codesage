"""Documentation command group for CodeSage CLI."""

from pathlib import Path
from typing import List, Optional

import typer

from codesage.cli.utils.console import get_console, print_error, print_warning, print_success
from codesage.cli.utils.decorators import handle_errors

app = typer.Typer(help="Generate documentation from indexed context")


@app.command("generate")
@handle_errors
def generate(
    path: str = typer.Argument(".", help="Project directory"),
    output: str = typer.Option(None, "--output", "-o", help="Output directory"),
    sections: str = typer.Option(
        "",
        "--sections",
        help="Comma-separated sections (overview, architecture, patterns, common_tasks)",
    ),
    include_onboarding: bool = typer.Option(
        False,
        "--include-onboarding",
        help="Also generate an onboarding guide",
    ),
) -> None:
    """Generate documentation into a docs folder.

    Use --include-onboarding to also generate a getting started guide
    for new developers.
    """
    from codesage.utils.config import Config
    from codesage.docs.generator import DocumentationGenerator

    console = get_console()
    project_path = Path(path).resolve()

    try:
        config = Config.load(project_path)
    except FileNotFoundError:
        print_error("Project not initialized.")
        raise typer.Exit(1)

    if not config.features.docs_generation:
        print_warning("Documentation generation is disabled.")
        console.print("  Enable with: [cyan]codesage features enable docs_generation[/cyan]")
        raise typer.Exit(1)

    generator = DocumentationGenerator(config)

    output_dir = Path(output) if output else project_path / config.docs.output_dir
    section_list = [s.strip() for s in sections.split(",") if s.strip()] or None

    out_file = generator.write(output_dir, sections=section_list)
    print_success(f"Wrote documentation to {out_file}")

    # Generate onboarding guide if requested
    if include_onboarding:
        from codesage.docs.onboarding import OnboardingGuideGenerator

        onboarding_generator = OnboardingGuideGenerator(config)
        onboarding_file = onboarding_generator.write(output_dir)
        print_success(f"Wrote onboarding guide to {onboarding_file}")


@app.command("onboarding")
@handle_errors
def onboarding(
    path: str = typer.Argument(".", help="Project directory"),
    output: str = typer.Option(None, "--output", "-o", help="Output directory"),
) -> None:
    """Generate onboarding guide into a docs folder.

    Note: This command is deprecated. Use 'codesage docs generate --include-onboarding' instead.
    """
    from codesage.utils.config import Config
    from codesage.docs.onboarding import OnboardingGuideGenerator

    console = get_console()
    project_path = Path(path).resolve()

    # Show deprecation notice
    print_warning(
        "The 'docs onboarding' command is deprecated. "
        "Use 'codesage docs generate --include-onboarding' instead."
    )

    try:
        config = Config.load(project_path)
    except FileNotFoundError:
        print_error("Project not initialized.")
        raise typer.Exit(1)

    if not config.features.docs_generation:
        print_warning("Documentation generation is disabled.")
        console.print("  Enable with: [cyan]codesage features enable docs_generation[/cyan]")
        raise typer.Exit(1)

    generator = OnboardingGuideGenerator(config)
    output_dir = Path(output) if output else project_path / config.docs.output_dir
    out_file = generator.write(output_dir)
    print_success(f"Wrote onboarding guide to {out_file}")

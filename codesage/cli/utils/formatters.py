"""Shared formatting utilities for CodeSage CLI output."""

from typing import Optional


def format_similarity(value: float) -> str:
    """Format similarity score as percentage.

    Args:
        value: Similarity score between 0 and 1.

    Returns:
        Formatted percentage string (e.g., "65%").
    """
    return f"{value:.0%}"


def format_risk_score(value: float) -> str:
    """Format risk score with colored indicator.

    Args:
        value: Risk score between 0 and 1.

    Returns:
        Rich-formatted risk indicator string.
    """
    if value >= 0.8:
        return f"[red bold]{value:.2f} (critical)[/red bold]"
    elif value >= 0.6:
        return f"[red]{value:.2f} (high)[/red]"
    elif value >= 0.4:
        return f"[yellow]{value:.2f} (medium)[/yellow]"
    elif value >= 0.2:
        return f"[blue]{value:.2f} (low)[/blue]"
    else:
        return f"[green]{value:.2f} (minimal)[/green]"


def severity_icon(level: str) -> str:
    """Get consistent icon for severity level.

    Args:
        level: Severity level string (error, warning, info, etc.).

    Returns:
        Rich-formatted icon string.
    """
    level_lower = level.lower()
    icons = {
        "critical": "[red bold]!![/red bold]",
        "error": "[red]![/red]",
        "warning": "[yellow]![/yellow]",
        "info": "[blue]i[/blue]",
        "hint": "[dim].[/dim]",
    }
    return icons.get(level_lower, "[dim]?[/dim]")


def format_count_summary(
    total: int,
    errors: int = 0,
    warnings: int = 0,
    info: int = 0,
    item_name: str = "issues",
) -> str:
    """Format a summary of counts by severity.

    Args:
        total: Total count.
        errors: Number of errors.
        warnings: Number of warnings.
        info: Number of info items.
        item_name: Name of the items being counted.

    Returns:
        Rich-formatted summary string.
    """
    parts = [f"Found [bold]{total}[/bold] {item_name}"]
    breakdown = []

    if errors > 0:
        breakdown.append(f"[red]{errors} error{'s' if errors != 1 else ''}[/red]")
    if warnings > 0:
        breakdown.append(f"[yellow]{warnings} warning{'s' if warnings != 1 else ''}[/yellow]")
    if info > 0:
        breakdown.append(f"[blue]{info} info[/blue]")

    if breakdown:
        parts.append(f": {', '.join(breakdown)}")

    return "".join(parts)


def format_file_location(file: str, line: Optional[int] = None) -> str:
    """Format file location consistently.

    Args:
        file: File path.
        line: Optional line number.

    Returns:
        Formatted location string.
    """
    if line is not None:
        return f"[bold blue]{file}:{line}[/bold blue]"
    return f"[bold blue]{file}[/bold blue]"


def format_impact_score(score: Optional[float]) -> str:
    """Format impact/blast radius score.

    Args:
        score: Impact score between 0 and 1, or None.

    Returns:
        Rich-formatted impact indicator.
    """
    if score is None:
        return "[dim]N/A[/dim]"

    if score >= 0.7:
        return f"[red bold]High ({score:.2f})[/red bold]"
    elif score >= 0.4:
        return f"[yellow]Medium ({score:.2f})[/yellow]"
    else:
        return f"[green]Low ({score:.2f})[/green]"


def truncate_code(code: str, max_length: int = 400) -> str:
    """Truncate code snippet with indicator.

    Args:
        code: Code string to truncate.
        max_length: Maximum length before truncation.

    Returns:
        Truncated code with indicator if needed.
    """
    if len(code) <= max_length:
        return code
    return code[:max_length] + "\n... [truncated]"

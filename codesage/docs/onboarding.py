"""Onboarding guide generator for CodeSage."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from codesage.core.context_provider import ContextProvider
from codesage.memory.memory_manager import MemoryManager
from codesage.utils.config import Config


class OnboardingGuideGenerator:
    """Generate onboarding guides from learned context."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self._context = ContextProvider(config)
        self._memory = None
        if config.memory.enabled:
            self._memory = MemoryManager(global_dir=config.memory.global_dir)

    def generate(self, tasks: Optional[List[str]] = None) -> str:
        if tasks is None:
            tasks = [
                "Set up the project",
                "Add a new feature",
                "Write tests",
            ]

        parts: List[str] = [
            "# Onboarding Guide",
            "",
            f"Project: {self.config.project_name}",
            "",
        ]

        parts.append("## Getting Started")
        parts.append("1. Run `codesage init` in the project root.")
        parts.append("2. Run `codesage index` to build context.")
        parts.append("3. Use `codesage context \"task\"` for guidance.")
        parts.append("")

        if self._memory:
            patterns = self._memory.preference_store.get_patterns(limit=5)
            parts.append("## Key Patterns")
            if patterns:
                for p in patterns:
                    parts.append(f"- {p.name} (confidence: {p.confidence_score:.2f})")
            else:
                parts.append("No learned patterns available yet.")
            parts.append("")

        parts.append("## Common Tasks")
        for task in tasks:
            ctx = self._context.get_implementation_context(task)
            parts.append(f"### {task}")
            for i, step in enumerate(ctx.implementation_plan.steps[:3], 1):
                parts.append(f"{i}. {step}")
            parts.append("")

        return "\n".join(parts).strip() + "\n"

    def write(self, output_dir: Path, tasks: Optional[List[str]] = None) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        content = self.generate(tasks=tasks)
        out_file = output_dir / "onboarding.md"
        out_file.write_text(content, encoding="utf-8")
        return out_file

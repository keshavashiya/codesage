"""Pattern deviation based code smell detection."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from codesage.models.code_element import CodeElement
from codesage.models.smell import CodeSmell
from codesage.memory.style_analyzer import StyleAnalyzer
from codesage.storage.database import Database
from codesage.utils.config import Config
from codesage.utils.logging import get_logger

logger = get_logger("review.smells")


@dataclass
class _BaselineStats:
    naming_patterns: Dict[str, Tuple[str, float]]
    error_handling_rate: Dict[str, float]
    avg_length: Dict[str, float]
    std_length: Dict[str, float]


class PatternDeviationDetector:
    """Detect code smells by comparing elements to learned baselines."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self._db = Database(config.storage.db_path)
        self._style = StyleAnalyzer()
        self._baseline: Optional[_BaselineStats] = None

    def detect_smells(
        self,
        elements: List[CodeElement],
        threshold: float = 0.7,
    ) -> List[CodeSmell]:
        """Detect smells in a list of code elements."""
        if not elements:
            return []

        baseline = self._get_baseline()
        smells: List[CodeSmell] = []

        for element in elements:
            smells.extend(self._check_naming(element, baseline, threshold))
            smells.extend(self._check_error_handling(element, baseline, threshold))
            smells.extend(self._check_complexity(element, baseline))
            smells.extend(self._check_security(element))

        return smells

    def detect_file(
        self,
        file_path: Path,
        threshold: float = 0.7,
    ) -> List[CodeSmell]:
        elements = self._db.get_elements_for_file(str(file_path))
        return self.detect_smells(elements, threshold=threshold)

    def _get_baseline(self) -> _BaselineStats:
        if self._baseline is not None:
            return self._baseline

        naming_counts: Dict[str, Dict[str, int]] = {}
        naming_totals: Dict[str, int] = {}
        error_counts: Dict[str, int] = {}
        error_totals: Dict[str, int] = {}
        lengths: Dict[str, List[int]] = {}

        max_samples = 1000
        sample_count = 0

        for element in self._db.get_all_elements(batch_size=200):
            if sample_count >= max_samples:
                break
            sample_count += 1

            element_type = element.type
            name = element.name or ""

            if name:
                naming_totals[element_type] = naming_totals.get(element_type, 0) + 1
                for pname, cfg in self._style.NAMING_PATTERNS.items():
                    if element_type in cfg["applies_to"]:
                        if re.match(cfg["pattern"], name):
                            naming_counts.setdefault(element_type, {})
                            naming_counts[element_type][pname] = (
                                naming_counts[element_type].get(pname, 0) + 1
                            )

            if element_type in ("function", "method"):
                error_totals[element_type] = error_totals.get(element_type, 0) + 1
                if "try:" in element.code and "except" in element.code:
                    error_counts[element_type] = error_counts.get(element_type, 0) + 1

                length = max(0, element.line_end - element.line_start + 1)
                lengths.setdefault(element_type, []).append(length)

        naming_patterns: Dict[str, Tuple[str, float]] = {}
        for element_type, total in naming_totals.items():
            if total == 0:
                continue
            best_pattern = None
            best_rate = 0.0
            for pname, count in naming_counts.get(element_type, {}).items():
                rate = count / total
                if rate > best_rate:
                    best_rate = rate
                    best_pattern = pname
            if best_pattern:
                naming_patterns[element_type] = (best_pattern, best_rate)

        error_rates = {}
        for element_type, total in error_totals.items():
            if total:
                error_rates[element_type] = error_counts.get(element_type, 0) / total

        avg_length = {}
        std_length = {}
        for element_type, values in lengths.items():
            if not values:
                continue
            avg = sum(values) / len(values)
            variance = sum((v - avg) ** 2 for v in values) / len(values)
            avg_length[element_type] = avg
            std_length[element_type] = math.sqrt(variance)

        self._baseline = _BaselineStats(
            naming_patterns=naming_patterns,
            error_handling_rate=error_rates,
            avg_length=avg_length,
            std_length=std_length,
        )
        return self._baseline

    def _check_naming(
        self,
        element: CodeElement,
        baseline: _BaselineStats,
        threshold: float,
    ) -> List[CodeSmell]:
        if not element.name:
            return []

        entry = baseline.naming_patterns.get(element.type)
        if not entry:
            return []

        pattern_name, rate = entry
        if rate < threshold:
            return []

        pattern_cfg = self._style.NAMING_PATTERNS.get(pattern_name)
        if not pattern_cfg:
            return []

        if not re.match(pattern_cfg["pattern"], element.name):
            return [
                CodeSmell(
                    type="naming_convention",
                    file=str(element.file),
                    line=element.line_start,
                    element=element.name,
                    severity="info",
                    confidence=rate,
                    message=(
                        f"Name deviates from dominant pattern '{pattern_name}' "
                        f"({rate:.0%} of similar elements)."
                    ),
                    suggestion=f"Use pattern: {pattern_cfg['description']}",
                    pattern=pattern_name,
                )
            ]
        return []

    def _check_error_handling(
        self,
        element: CodeElement,
        baseline: _BaselineStats,
        threshold: float,
    ) -> List[CodeSmell]:
        if element.type not in ("function", "method"):
            return []

        rate = baseline.error_handling_rate.get(element.type, 0.0)
        if rate < threshold:
            return []

        has_error_handling = "try:" in element.code and "except" in element.code
        if not has_error_handling:
            return [
                CodeSmell(
                    type="missing_error_handling",
                    file=str(element.file),
                    line=element.line_start,
                    element=element.name,
                    severity="warning",
                    confidence=rate,
                    message=(
                        f"{rate:.0%} of similar functions use try/except, but this doesn't."
                    ),
                    suggestion="Add error handling consistent with project patterns.",
                    pattern="error_handling",
                )
            ]
        return []

    def _check_complexity(
        self,
        element: CodeElement,
        baseline: _BaselineStats,
    ) -> List[CodeSmell]:
        if element.type not in ("function", "method"):
            return []

        avg = baseline.avg_length.get(element.type)
        std = baseline.std_length.get(element.type, 0.0)
        if avg is None:
            return []

        length = max(0, element.line_end - element.line_start + 1)
        if std > 0 and length > avg + 2 * std and length > 40:
            return [
                CodeSmell(
                    type="complexity_spike",
                    file=str(element.file),
                    line=element.line_start,
                    element=element.name,
                    severity="warning",
                    confidence=0.6,
                    message=(
                        f"Function length ({length} lines) is much larger than "
                        f"average ({avg:.0f})."
                    ),
                    suggestion="Consider refactoring into smaller helpers.",
                )
            ]
        return []

    def _check_security(self, element: CodeElement) -> List[CodeSmell]:
        code = element.code.lower()
        smells: List[CodeSmell] = []

        if "password" in code and not any(k in code for k in ["hash", "bcrypt", "scrypt"]):
            smells.append(
                CodeSmell(
                    type="password_handling",
                    file=str(element.file),
                    line=element.line_start,
                    element=element.name,
                    severity="error",
                    confidence=0.55,
                    message="Password handling lacks evidence of hashing.",
                    suggestion="Use a secure hashing function for passwords.",
                )
            )

        if "token" in code and not any(k in code for k in ["verify", "validate", "decode"]):
            smells.append(
                CodeSmell(
                    type="token_validation",
                    file=str(element.file),
                    line=element.line_start,
                    element=element.name,
                    severity="warning",
                    confidence=0.5,
                    message="Token usage without obvious validation or decoding.",
                    suggestion="Validate or decode tokens before use.",
                )
            )

        return smells

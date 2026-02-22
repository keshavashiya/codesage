"""Unified review pipeline orchestrator.

Combines all review sources: security scanner, static checks (bad practices,
complexity, structure, naming), pattern deviation, semantic similarity, and
optional LLM synthesis into a single review pass.

Two modes:
- fast: Static checks only (security + AST analysis). Target: <5 seconds.
- full: Everything including semantic similarity search and LLM synthesis.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Set

from codesage.review.models import (
    FileChange,
    ReviewFinding,
    UnifiedReviewResult,
)
from codesage.review.suppression import (
    ProjectSuppressions,
    SuppressionParser,
    apply_suppressions,
)
from codesage.utils.language_detector import GENERIC_EXTENSIONS
from codesage.utils.logging import get_logger

logger = get_logger(__name__)


class ReviewPipeline:
    """Orchestrates all review checks into a single pipeline.

    Usage::

        pipeline = ReviewPipeline(repo_path=Path("."))
        result = pipeline.run()
        if result.has_blocking_issues("high"):
            sys.exit(1)
    """

    def __init__(
        self,
        repo_path: Path,
        config=None,
        mode: str = "fast",
    ):
        self.repo_path = repo_path.resolve()
        self.config = config
        self.mode = mode

        # Lazy-loaded components
        self._diff_extractor = None
        self._security_scanner = None
        self._suppression_parser = SuppressionParser()
        self._project_suppressions = None

    @property
    def diff_extractor(self):
        if self._diff_extractor is None:
            from codesage.review.diff import DiffExtractor
            self._diff_extractor = DiffExtractor(self.repo_path)
        return self._diff_extractor

    @property
    def security_scanner(self):
        if self._security_scanner is None:
            from codesage.security.scanner import SecurityScanner
            from codesage.security.models import Severity
            self._security_scanner = SecurityScanner(
                severity_threshold=Severity.LOW,
                include_tests=False,
            )
        return self._security_scanner

    @property
    def project_suppressions(self) -> ProjectSuppressions:
        if self._project_suppressions is None:
            self._project_suppressions = ProjectSuppressions.load(self.repo_path)
        return self._project_suppressions

    def run(
        self,
        staged_only: bool = False,
        severity_threshold: str = "suggestion",
    ) -> UnifiedReviewResult:
        """Execute the full review pipeline.

        Args:
            staged_only: Only review staged changes (for pre-commit).
            severity_threshold: Minimum severity for blocking (used by caller).

        Returns:
            UnifiedReviewResult with all findings.
        """
        t_start = time.time()
        stage_timings: Dict[str, float] = {}
        stages_skipped: List[str] = []
        all_findings: List[ReviewFinding] = []

        # Step 1: Get changes
        t0 = time.time()
        try:
            if staged_only:
                changes = self.diff_extractor.get_staged_changes()
            else:
                changes = self.diff_extractor.get_all_changes()
        except Exception as e:
            logger.error(f"Failed to get git changes: {e}")
            return UnifiedReviewResult(
                summary=f"Error: Could not read git changes: {e}",
                mode=self.mode,
                duration_ms=(time.time() - t_start) * 1000,
            )
        stage_timings["diff"] = (time.time() - t0) * 1000

        if not changes:
            return UnifiedReviewResult(
                summary="No uncommitted changes found",
                mode=self.mode,
                duration_ms=(time.time() - t_start) * 1000,
                stage_timings=stage_timings,
            )

        # Filter out files excluded by .codesageignore before running any checks
        changes = [
            c for c in changes
            if not self.project_suppressions.is_file_excluded(c.path)
        ]

        if not changes:
            return UnifiedReviewResult(
                summary="No uncommitted changes found (all files excluded)",
                mode=self.mode,
                duration_ms=(time.time() - t_start) * 1000,
                stage_timings=stage_timings,
            )

        # Collect inline suppressions for changed files (all languages)
        file_suppressions: Dict[Path, Dict] = {}
        for change in changes:
            full_path = self.repo_path / change.path
            if full_path.exists() and full_path.is_file():
                supps = self._suppression_parser.parse_file(full_path)
                if supps:
                    file_suppressions[change.path] = supps

        # Step 2: Security scanning
        t0 = time.time()
        security_findings = self._run_security(changes)
        all_findings.extend(security_findings)
        stage_timings["security"] = (time.time() - t0) * 1000

        # Step 3: Static AST checks (Python only)
        t0 = time.time()
        static_findings = self._run_static_checks(changes)
        all_findings.extend(static_findings)
        stage_timings["static_checks"] = (time.time() - t0) * 1000

        # Step 4: Pattern deviation (requires index)
        if self.mode == "full" or self._has_index():
            t0 = time.time()
            pattern_findings = self._run_pattern_checks(changes)
            if pattern_findings is not None:
                all_findings.extend(pattern_findings)
                stage_timings["patterns"] = (time.time() - t0) * 1000
            else:
                stages_skipped.append("patterns")
        else:
            stages_skipped.append("patterns")

        # Step 5: Semantic similarity (full mode + index required)
        if self.mode == "full" and self._has_index():
            t0 = time.time()
            similarity_findings = self._run_similarity(changes)
            if similarity_findings is not None:
                all_findings.extend(similarity_findings)
                stage_timings["similarity"] = (time.time() - t0) * 1000
            else:
                stages_skipped.append("similarity")
        else:
            stages_skipped.append("similarity")

        # Step 6: LLM synthesis (full mode only)
        if self.mode == "full":
            t0 = time.time()
            llm_findings = self._run_llm_synthesis(changes, all_findings)
            if llm_findings is not None:
                all_findings.extend(llm_findings)
                stage_timings["llm"] = (time.time() - t0) * 1000
            else:
                stages_skipped.append("llm")
        else:
            stages_skipped.append("llm")

        # Step 7: Dedup PAT-COMPLEXITY-SPIKE where PY-LONG-FUNCTION already fired
        long_fn_locs: set = {
            (str(f.file), f.line) for f in all_findings
            if f.rule_id == "PY-LONG-FUNCTION"
        }
        all_findings = [
            f for f in all_findings
            if not (
                f.rule_id == "PAT-COMPLEXITY-SPIKE"
                and (str(f.file), f.line) in long_fn_locs
            )
        ]

        # Step 8: Apply suppressions
        apply_suppressions(all_findings, file_suppressions, self.project_suppressions)

        # Build result
        duration_ms = (time.time() - t_start) * 1000
        result = UnifiedReviewResult(
            findings=all_findings,
            files_changed=changes,
            mode=self.mode,
            duration_ms=duration_ms,
            stage_timings=stage_timings,
            stages_skipped=stages_skipped,
        )
        result.summary = self._build_summary(result)

        return result

    def run_on_file(self, file_path: Path) -> "UnifiedReviewResult":
        """Run the review pipeline on a single file (not git-diff-based).

        Creates a synthetic FileChange for the given file and runs security
        and static checks (always), plus pattern/similarity if index is present.

        Args:
            file_path: Absolute or repo-relative path to the file.

        Returns:
            UnifiedReviewResult with all findings for that file.
        """
        t_start = time.time()
        stage_timings: Dict[str, float] = {}
        stages_skipped: List[str] = []
        all_findings: List[ReviewFinding] = []

        # Resolve path
        if not file_path.is_absolute():
            file_path = self.repo_path / file_path
        if not file_path.exists():
            return UnifiedReviewResult(
                summary=f"File not found: {file_path}",
                mode=self.mode,
                duration_ms=(time.time() - t_start) * 1000,
            )

        rel_path = file_path.relative_to(self.repo_path) if file_path.is_relative_to(self.repo_path) else file_path
        changes = [FileChange(path=rel_path, status="M")]

        # Inline suppressions for this file
        file_suppressions: Dict[Path, Dict] = {}
        if file_path.suffix == ".py":
            supps = self._suppression_parser.parse_file(file_path)
            if supps:
                file_suppressions[rel_path] = supps

        # Security
        t0 = time.time()
        all_findings.extend(self._run_security(changes))
        stage_timings["security"] = (time.time() - t0) * 1000

        # Static checks
        t0 = time.time()
        all_findings.extend(self._run_static_checks(changes))
        stage_timings["static_checks"] = (time.time() - t0) * 1000

        # Pattern deviation (if index available)
        if self._has_index():
            t0 = time.time()
            pattern_findings = self._run_pattern_checks(changes)
            if pattern_findings is not None:
                all_findings.extend(pattern_findings)
                stage_timings["patterns"] = (time.time() - t0) * 1000
            else:
                stages_skipped.append("patterns")
        else:
            stages_skipped.append("patterns")

        # Similarity (full mode only)
        if self.mode == "full" and self._has_index():
            t0 = time.time()
            sim_findings = self._run_similarity(changes)
            if sim_findings is not None:
                all_findings.extend(sim_findings)
                stage_timings["similarity"] = (time.time() - t0) * 1000
            else:
                stages_skipped.append("similarity")
        else:
            stages_skipped.append("similarity")

        # No LLM synthesis for single-file chat review to keep it fast
        stages_skipped.append("llm")

        apply_suppressions(all_findings, file_suppressions, self.project_suppressions)

        duration_ms = (time.time() - t_start) * 1000
        result = UnifiedReviewResult(
            findings=all_findings,
            files_changed=changes,
            mode=self.mode,
            duration_ms=duration_ms,
            stage_timings=stage_timings,
            stages_skipped=stages_skipped,
        )
        result.summary = self._build_summary(result)
        return result

    # --- Stage implementations ---

    def _run_security(self, changes: List[FileChange]) -> List[ReviewFinding]:
        """Run security scanner on changed files."""
        findings: List[ReviewFinding] = []

        for change in changes:
            if change.status == "D":
                continue

            full_path = self.repo_path / change.path
            if not full_path.exists():
                continue

            try:
                sec_findings = self.security_scanner.scan_file(full_path)
                for sf in sec_findings:
                    findings.append(ReviewFinding(
                        severity=sf.rule.severity.value,
                        category="security",
                        file=change.path,
                        line=sf.line_number,
                        rule_id=sf.rule.id,
                        message=sf.rule.message,
                        suggestion=sf.rule.fix_suggestion,
                        source="static",
                    ))
            except Exception as e:
                logger.warning(f"Security scan failed for {change.path}: {e}")

        return findings

    def _run_static_checks(self, changes: List[FileChange]) -> List[ReviewFinding]:
        """Run static checks on changed files.

        Python files: full AST-based analysis (bad practices, complexity,
        structure, naming).
        Non-Python files (Rust/Go/JS/TS): tree-sitter AST checks when available,
        falling back to language-agnostic text checks.
        """
        from codesage.review.checks import (
            PythonBadPracticeChecker,
            ComplexityChecker,
            StructureChecker,
            NamingChecker,
            GenericFileChecker,
        )
        from codesage.review.checks.treesitter_checks import (
            TreeSitterReviewChecker,
            TREESITTER_AVAILABLE,
        )

        _PYTHON_CHECKERS = [
            PythonBadPracticeChecker(),
            ComplexityChecker(),
            StructureChecker(),
            NamingChecker(),
        ]
        # Prefer tree-sitter for non-Python; fall back to generic text checks
        _NON_PY_CHECKER = (
            TreeSitterReviewChecker() if TREESITTER_AVAILABLE else GenericFileChecker()
        )

        findings: List[ReviewFinding] = []

        for change in changes:
            if change.status == "D":
                continue

            full_path = self.repo_path / change.path
            if not full_path.exists():
                continue

            ext = full_path.suffix.lower()

            try:
                content = full_path.read_text(errors="replace")
            except Exception:
                continue

            if ext == ".py":
                # Full Python AST checks
                for checker in _PYTHON_CHECKERS:
                    try:
                        file_findings = checker.check(change.path, content)
                        findings.extend(file_findings)
                    except Exception as e:
                        logger.warning(
                            f"{checker.__class__.__name__} failed for {change.path}: {e}"
                        )
            elif ext in GENERIC_EXTENSIONS:
                # Tree-sitter (preferred) or text-based checks for non-Python
                try:
                    file_findings = _NON_PY_CHECKER.check(full_path, content)
                    # Re-map file path to the relative change.path for consistency
                    for f in file_findings:
                        f.file = change.path
                    findings.extend(file_findings)
                except Exception as e:
                    logger.warning(
                        f"{_NON_PY_CHECKER.__class__.__name__} failed for {change.path}: {e}"
                    )

        return findings

    def _run_pattern_checks(self, changes: List[FileChange]) -> Optional[List[ReviewFinding]]:
        """Run pattern deviation detection using the codebase baseline.

        Compares changed files against learned codebase patterns (naming,
        error handling, complexity relative to project baseline).

        Returns None if the pattern checker or index is unavailable.
        """
        if not self._has_index() or not self.config:
            return None

        try:
            from codesage.review.smells import PatternDeviationDetector

            detector = PatternDeviationDetector(self.config)

            findings: List[ReviewFinding] = []
            for change in changes:
                if change.status == "D":
                    continue

                full_path = self.repo_path / change.path
                if not full_path.exists():
                    continue

                try:
                    # detect_file() fetches indexed elements for this file via DB
                    smells = detector.detect_file(full_path)
                    for smell in smells:
                        stype = getattr(smell, "type", "deviation")
                        rule_id = f"PAT-{stype.upper().replace(' ', '-').replace('_', '-')}"
                        findings.append(ReviewFinding(
                            severity=_map_smell_severity(getattr(smell, "severity", "info")),
                            category="pattern",
                            file=change.path,
                            line=getattr(smell, "line", None),
                            rule_id=rule_id,
                            message=getattr(smell, "message", str(smell)),
                            suggestion=getattr(smell, "suggestion", None),
                            source="static",
                        ))
                except Exception as e:
                    logger.debug(f"Pattern check failed for {change.path}: {e}")

            return findings
        except Exception as e:
            logger.debug(f"Pattern deviation detection unavailable: {e}")
            return None

    def _run_similarity(self, changes: List[FileChange]) -> Optional[List[ReviewFinding]]:
        """Run semantic similarity search against indexed codebase.

        Returns None if the vector store is unavailable.
        """
        if not self.config:
            return None

        try:
            from codesage.storage.manager import StorageManager
            from codesage.llm.embeddings import EmbeddingService

            embedder = EmbeddingService(
                self.config.llm,
                self.config.cache_dir,
                self.config.performance,
            )
            storage = StorageManager(config=self.config, embedding_fn=embedder)

            if not storage.vector_store:
                return None

            findings: List[ReviewFinding] = []
            for change in changes:
                if change.status == "D" or not change.diff:
                    continue

                # Extract added lines
                added_lines = []
                for line in change.diff.split("\n"):
                    if line.startswith("+") and not line.startswith("+++"):
                        added_lines.append(line[1:])
                added_code = "\n".join(added_lines)

                if len(added_code) < 50:
                    continue

                try:
                    results = storage.vector_store.query(
                        query_text=added_code[:1500],
                        n_results=3,
                    )
                    for match in results:
                        similarity = match.get("similarity", 0)
                        metadata = match.get("metadata", {})
                        match_file = metadata.get("file", "")

                        # Skip self-matches
                        if str(change.path) in match_file:
                            continue

                        if similarity >= 0.85:
                            findings.append(ReviewFinding(
                                severity="warning",
                                category="duplication",
                                file=change.path,
                                rule_id="DUP-HIGH",
                                message=f"Possible duplicate of '{metadata.get('name', '?')}' in {match_file} ({similarity:.0%} similar)",
                                suggestion="Consider reusing the existing code instead of duplicating",
                                source="semantic",
                            ))
                        elif similarity >= 0.70:
                            findings.append(ReviewFinding(
                                severity="suggestion",
                                category="duplication",
                                file=change.path,
                                rule_id="DUP-SIMILAR",
                                message=f"Similar to '{metadata.get('name', '?')}' in {match_file} ({similarity:.0%} similar)",
                                suggestion="Review existing code — you might extend it instead",
                                source="semantic",
                            ))
                except Exception as e:
                    logger.debug(f"Similarity search failed for {change.path}: {e}")

            return findings
        except Exception as e:
            logger.debug(f"Similarity search unavailable: {e}")
            return None

    def _run_llm_synthesis(
        self,
        changes: List[FileChange],
        existing_findings: List[ReviewFinding],
    ) -> Optional[List[ReviewFinding]]:
        """Use LLM to synthesize additional insights from changes.

        Returns None if LLM is unavailable.
        """
        if not self.config:
            return None

        try:
            from codesage.llm.provider import LLMProvider
            llm = LLMProvider(self.config.llm)
        except Exception as e:
            logger.debug(f"LLM unavailable for synthesis: {e}")
            return None

        # Only send files that have issues or are new
        files_with_issues: Set[str] = {str(f.file) for f in existing_findings}
        review_changes = [
            c for c in changes
            if c.status != "D" and (
                str(c.path) in files_with_issues or c.status == "A"
            )
        ][:5]  # Limit to 5 files

        if not review_changes:
            return []

        # Build prompt
        prompt_parts = [
            "Review these code changes for issues NOT already caught by static analysis.\n",
            "Focus on: logic errors, missing edge cases, performance issues.\n\n",
        ]

        for change in review_changes:
            diff_preview = change.diff[:2000] if change.diff else "(no diff)"
            prompt_parts.append(f"## {change.path} ({change.status})\n```\n{diff_preview}\n```\n\n")

        existing_by_file: Dict[str, List[str]] = {}
        for f in existing_findings:
            key = str(f.file)
            if key not in existing_by_file:
                existing_by_file[key] = []
            existing_by_file[key].append(f.message)

        if existing_by_file:
            prompt_parts.append("Already found (do NOT repeat):\n")
            for fpath, msgs in list(existing_by_file.items())[:10]:
                for msg in msgs[:3]:
                    prompt_parts.append(f"- {fpath}: {msg}\n")

        prompt_parts.append(
            "\nProvide ONLY new findings.\n"
            "Format: [SEVERITY] file:line - description\n"
            "Severities: CRITICAL, HIGH, WARNING, SUGGESTION\n"
        )

        try:
            messages = [
                {"role": "system", "content": "You are a senior code reviewer. Be concise. Only report meaningful issues."},
                {"role": "user", "content": "".join(prompt_parts)},
            ]
            response = llm.chat(messages)

            # Parse response
            findings: List[ReviewFinding] = []
            import re
            for line in response.strip().split("\n"):
                line = line.strip()
                match = re.match(
                    r"\[?(CRITICAL|HIGH|WARNING|SUGGESTION)\]?\s*(.+?)(?::(\d+))?\s*[-–]\s*(.+)",
                    line, re.IGNORECASE,
                )
                if match:
                    severity = match.group(1).lower()
                    file_path = match.group(2).strip()
                    line_num = int(match.group(3)) if match.group(3) else None
                    message = match.group(4).strip()

                    findings.append(ReviewFinding(
                        severity=severity,
                        category="review",
                        file=Path(file_path),
                        line=line_num,
                        message=message,
                        source="llm",
                    ))

            return findings
        except Exception as e:
            logger.warning(f"LLM synthesis failed: {e}")
            return None

    # --- Helpers ---

    def _has_index(self) -> bool:
        """Check if the project has been indexed."""
        db_path = self.repo_path / ".codesage" / "codesage.db"
        return db_path.exists()

    def _build_summary(self, result: UnifiedReviewResult) -> str:
        """Build a human-readable summary line."""
        parts = []

        n_files = len(result.files_changed)
        parts.append(f"{n_files} file{'s' if n_files != 1 else ''} changed")
        parts.append(f"+{result.total_additions} -{result.total_deletions}")

        counts = []
        if result.critical_count:
            counts.append(f"{result.critical_count} critical")
        if result.high_count:
            counts.append(f"{result.high_count} high")
        if result.warning_count:
            counts.append(f"{result.warning_count} warning{'s' if result.warning_count != 1 else ''}")
        if result.suggestion_count:
            counts.append(f"{result.suggestion_count} suggestion{'s' if result.suggestion_count != 1 else ''}")
        if result.praise_count:
            counts.append(f"{result.praise_count} good practice{'s' if result.praise_count != 1 else ''}")

        if counts:
            parts.append(", ".join(counts))
        else:
            parts.append("no issues found")

        if result.suppressed_count:
            parts.append(f"({result.suppressed_count} suppressed)")

        return " | ".join(parts)


def _map_smell_severity(smell_severity: str) -> str:
    """Map CodeSmell severity to ReviewFinding severity."""
    mapping = {
        "error": "high",
        "warning": "warning",
        "info": "suggestion",
        "suggestion": "suggestion",
    }
    return mapping.get(smell_severity.lower(), "suggestion")

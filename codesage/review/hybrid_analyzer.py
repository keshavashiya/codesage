"""Hybrid code review analyzer.

Uses a multi-stage pipeline:
1. Static analysis (security patterns, anti-patterns)
2. Semantic duplicate detection (vector search against codebase)
3. LLM synthesis (final verdict and suggestions based on findings)

This approach is faster, cheaper, and more accurate than pure LLM review.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any

from codesage.review.models import FileChange, IssueSeverity, ReviewIssue, ReviewResult
from codesage.review.diff import DiffExtractor
from codesage.security import SecurityScanner
from codesage.security.models import Severity
from codesage.utils.config import Config
from codesage.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SimilarCode:
    """Represents similar code found in the codebase."""

    file: Path
    line: int
    name: str
    similarity: float
    code_snippet: str
    element_type: str


@dataclass
class AnalysisContext:
    """Context gathered from static analysis for LLM synthesis."""

    file_path: Path
    diff_content: str
    security_issues: List[Dict[str, Any]] = field(default_factory=list)
    similar_code: List[SimilarCode] = field(default_factory=list)
    impact_callers: List[Dict[str, Any]] = field(default_factory=list)  # Functions that call changed code
    is_new_file: bool = False

    @property
    def has_issues(self) -> bool:
        return len(self.security_issues) > 0 or len(self.similar_code) > 0

    @property
    def has_impact(self) -> bool:
        return len(self.impact_callers) > 0


class HybridReviewAnalyzer:
    """Hybrid code review using static analysis + semantic search + LLM.

    This analyzer:
    1. Runs security pattern matching (fast, no LLM)
    2. Searches for similar/duplicate code in the indexed codebase
    3. Only uses LLM to synthesize findings into human-readable feedback

    Benefits:
    - Finds actual duplicates in YOUR codebase (not general patterns)
    - Identifies security issues instantly (pattern matching)
    - LLM cost reduced by 80%+ (only used for synthesis)
    - Much faster review times
    """

    # Similarity threshold for duplicate detection
    DUPLICATE_THRESHOLD = 0.85
    SIMILAR_THRESHOLD = 0.70

    # Maximum diff size to send to LLM for synthesis
    MAX_CONTEXT_SIZE = 6000

    def __init__(
        self,
        config: Optional[Config] = None,
        repo_path: Optional[Path] = None,
    ):
        """Initialize the hybrid analyzer."""
        self.config = config
        self.repo_path = repo_path or Path.cwd()
        self._diff_extractor = DiffExtractor(self.repo_path)

        # Lazy initialization
        self._storage_manager = None
        self._llm = None
        self._security_scanner = None

    @property
    def storage(self):
        """Lazy load unified storage manager."""
        if self._storage_manager is None and self.config:
            from codesage.storage.manager import StorageManager
            from codesage.llm.embeddings import EmbeddingService

            embedder = EmbeddingService(self.config.llm, self.config.cache_dir)
            self._storage_manager = StorageManager(
                config=self.config,
                embedding_fn=embedder.embedder,
            )
        return self._storage_manager

    @property
    def vector_store(self):
        """Get vector store from storage manager."""
        return self.storage.vector_store if self.storage else None

    @property
    def graph_store(self):
        """Get graph store from storage manager."""
        return self.storage.graph_store if self.storage else None

    @property
    def llm(self):
        """Lazy load LLM provider."""
        if self._llm is None and self.config:
            from codesage.llm.provider import LLMProvider
            self._llm = LLMProvider(self.config.llm)
        return self._llm

    @property
    def security_scanner(self) -> SecurityScanner:
        """Lazy load security scanner."""
        if self._security_scanner is None:
            self._security_scanner = SecurityScanner(
                severity_threshold=Severity.MEDIUM,
                include_tests=False,
            )
        return self._security_scanner

    def get_staged_changes(self) -> List[FileChange]:
        """Get list of staged file changes."""
        return self._diff_extractor.get_staged_changes()

    def get_all_changes(self) -> List[FileChange]:
        """Get all uncommitted changes."""
        return self._diff_extractor.get_all_changes()

    def review_changes(
        self,
        changes: Optional[List[FileChange]] = None,
        generate_pr_description: bool = False,
        use_llm_synthesis: bool = True,
    ) -> ReviewResult:
        """Review code changes using hybrid analysis.

        Pipeline:
        1. Static security analysis (instant)
        2. Semantic duplicate detection against indexed codebase
        3. LLM synthesis of findings (optional)

        Args:
            changes: File changes to review
            generate_pr_description: Generate PR description
            use_llm_synthesis: Whether to use LLM for final synthesis

        Returns:
            ReviewResult with issues and summary
        """
        if changes is None:
            changes = self.get_staged_changes()

        if not changes:
            return ReviewResult(summary="No changes to review")

        result = ReviewResult(files_changed=changes)
        analysis_contexts: List[AnalysisContext] = []

        # Stage 1 & 2: Static analysis and similarity search per file
        for change in changes:
            if change.status == "D" or not change.diff:
                continue

            context = self._analyze_file(change)
            analysis_contexts.append(context)

            # Convert security findings to ReviewIssues immediately
            for sec_issue in context.security_issues:
                result.issues.append(ReviewIssue(
                    severity=self._map_security_severity(sec_issue.get("severity", "medium")),
                    file=change.path,
                    line=sec_issue.get("line"),
                    message=sec_issue.get("message", "Security issue detected"),
                    suggestion=sec_issue.get("suggestion"),
                ))

            # Add duplicate/similar code warnings
            for similar in context.similar_code:
                if similar.similarity >= self.DUPLICATE_THRESHOLD:
                    result.issues.append(ReviewIssue(
                        severity=IssueSeverity.WARNING,
                        file=change.path,
                        line=None,
                        message=f"Possible duplicate of `{similar.name}` in {similar.file}:{similar.line} ({similar.similarity:.0%} similar)",
                        suggestion=f"Consider reusing existing {similar.element_type} instead of duplicating code.",
                        code_snippet=similar.code_snippet[:200] + "..." if len(similar.code_snippet) > 200 else similar.code_snippet,
                    ))
                elif similar.similarity >= self.SIMILAR_THRESHOLD:
                    result.issues.append(ReviewIssue(
                        severity=IssueSeverity.SUGGESTION,
                        file=change.path,
                        line=None,
                        message=f"Similar to `{similar.name}` in {similar.file}:{similar.line} ({similar.similarity:.0%} similar)",
                        suggestion=f"Review existing {similar.element_type} - you might be able to extend it instead.",
                    ))

            # Add impact warnings from graph analysis
            if context.impact_callers:
                # Group callers by element
                impacted = {}
                for caller in context.impact_callers:
                    elem = caller.get("element", "unknown")
                    if elem not in impacted:
                        impacted[elem] = []
                    impacted[elem].append(caller)

                for elem, callers in impacted.items():
                    caller_names = [f"`{c.get('caller_name')}`" for c in callers[:3]]
                    caller_list = ", ".join(caller_names)
                    more = f" and {len(callers) - 3} more" if len(callers) > 3 else ""

                    result.issues.append(ReviewIssue(
                        severity=IssueSeverity.SUGGESTION,
                        file=change.path,
                        line=None,
                        message=f"Impact: `{elem}` is called by {caller_list}{more}",
                        suggestion=f"Verify changes don't break these {len(callers)} caller(s).",
                    ))

        # Stage 3: LLM synthesis (optional, for additional insights)
        if use_llm_synthesis and self.llm:
            # Only send files with issues or new files for deeper review
            files_for_llm = [
                ctx for ctx in analysis_contexts
                if ctx.has_issues or ctx.is_new_file
            ][:5]  # Limit to 5 files for LLM review

            if files_for_llm:
                llm_issues = self._llm_synthesize(files_for_llm)
                result.issues.extend(llm_issues)

        # Generate summary
        result.summary = self._generate_summary(result, analysis_contexts)

        if generate_pr_description:
            result.pr_description = self._generate_pr_description(changes, analysis_contexts)

        return result

    def _analyze_file(self, change: FileChange) -> AnalysisContext:
        """Analyze a single file with static analysis and similarity search."""
        context = AnalysisContext(
            file_path=change.path,
            diff_content=change.diff[:self.MAX_CONTEXT_SIZE],
            is_new_file=(change.status == "A"),
        )

        # Stage 1: Security pattern matching
        full_path = self.repo_path / change.path
        if full_path.exists():
            try:
                findings = self.security_scanner.scan_file(full_path)
                context.security_issues = [
                    {
                        "severity": f.rule.severity.value,
                        "line": f.line_number,
                        "message": f.rule.message,
                        "suggestion": f.rule.fix_suggestion,
                        "rule_id": f.rule.id,
                    }
                    for f in findings
                ]
            except Exception as e:
                logger.warning(f"Security scan failed for {change.path}: {e}")

        # Stage 2: Semantic similarity search against indexed codebase
        if self.vector_store:
            try:
                # Extract meaningful code snippets from diff (added lines)
                added_lines = self._extract_added_code(change.diff)

                if added_lines and len(added_lines) > 50:  # Only search if substantial code
                    similar = self.vector_store.query(
                        query_text=added_lines[:1500],  # Truncate for embedding
                        n_results=3,
                    )

                    for match in similar:
                        similarity = match.get("similarity", 0)
                        if similarity >= self.SIMILAR_THRESHOLD:
                            metadata = match.get("metadata", {})
                            # Skip matches from the same file
                            match_file = metadata.get("file", "")
                            if str(change.path) in match_file:
                                continue

                            context.similar_code.append(SimilarCode(
                                file=Path(match_file),
                                line=metadata.get("line_start", 0),
                                name=metadata.get("name", "unknown"),
                                similarity=similarity,
                                code_snippet=match.get("document", "")[:500],
                                element_type=metadata.get("type", "code"),
                            ))
            except Exception as e:
                logger.warning(f"Similarity search failed for {change.path}: {e}")

        # Stage 3: Graph-based impact analysis (find callers of changed code)
        if self.graph_store and not context.is_new_file:
            try:
                context.impact_callers = self._analyze_impact(change)
            except Exception as e:
                logger.warning(f"Impact analysis failed for {change.path}: {e}")

        return context

    def _analyze_impact(self, change: FileChange) -> List[Dict[str, Any]]:
        """Analyze downstream impact of changes using graph store.

        Finds functions that call elements in the changed file.
        """
        if not self.graph_store:
            return []

        callers = []

        # Get elements defined in this file from the database
        if self.storage and self.storage.db:
            try:
                elements = self.storage.db.get_elements_for_file(str(change.path))

                for element in elements[:10]:  # Limit to first 10 elements
                    element_id = element.get("id", "")
                    if element_id:
                        # Find who calls this element
                        element_callers = self.graph_store.get_callers(element_id)

                        for caller in element_callers[:5]:  # Limit callers per element
                            # Don't include callers from the same file
                            caller_file = caller.get("file", "")
                            if str(change.path) not in caller_file:
                                callers.append({
                                    "element": element.get("name", element_id),
                                    "caller_name": caller.get("name", "unknown"),
                                    "caller_file": caller_file,
                                    "caller_line": caller.get("line_start", 0),
                                })
            except Exception as e:
                logger.debug(f"Could not get elements for impact analysis: {e}")

        return callers

    def _extract_added_code(self, diff: str) -> str:
        """Extract only added lines from a diff."""
        added_lines = []
        for line in diff.split("\n"):
            if line.startswith("+") and not line.startswith("+++"):
                added_lines.append(line[1:])  # Remove the + prefix
        return "\n".join(added_lines)

    def _map_security_severity(self, severity_value: str) -> IssueSeverity:
        """Map security severity to review severity."""
        mapping = {
            "critical": IssueSeverity.CRITICAL,
            "high": IssueSeverity.CRITICAL,
            "medium": IssueSeverity.WARNING,
            "low": IssueSeverity.SUGGESTION,
            "info": IssueSeverity.SUGGESTION,
        }
        return mapping.get(severity_value.lower(), IssueSeverity.WARNING)

    def _llm_synthesize(self, contexts: List[AnalysisContext]) -> List[ReviewIssue]:
        """Use LLM to synthesize findings into additional insights.

        This is only called for files with issues or new files,
        and the LLM is provided with the analysis context rather than
        reviewing the code from scratch.
        """
        issues = []

        prompt = self._build_synthesis_prompt(contexts)

        try:
            response = self.llm.generate(
                prompt=prompt,
                system_prompt=SYNTHESIS_SYSTEM_PROMPT,
                temperature=0.3,
            )

            # Parse LLM response
            issues = self._parse_llm_response(response, contexts)

        except Exception as e:
            logger.error(f"LLM synthesis failed: {e}")

        return issues

    def _build_synthesis_prompt(self, contexts: List[AnalysisContext]) -> str:
        """Build prompt for LLM synthesis based on gathered context."""
        prompt_parts = [
            "Based on the following analysis of code changes, provide additional insights.\n",
            "Focus on: logic errors, missing edge cases, performance issues, and best practices.\n",
            "Do NOT repeat the security issues or duplicate warnings already identified.\n\n"
        ]

        for ctx in contexts:
            prompt_parts.append(f"## File: {ctx.file_path}\n")
            prompt_parts.append(f"Status: {'New file' if ctx.is_new_file else 'Modified'}\n")

            if ctx.security_issues:
                prompt_parts.append(f"Already found {len(ctx.security_issues)} security issues\n")
            if ctx.similar_code:
                prompt_parts.append(f"Found {len(ctx.similar_code)} similar code patterns\n")

            # Add truncated diff
            diff_preview = ctx.diff_content[:2000]
            prompt_parts.append(f"\nCode changes:\n```\n{diff_preview}\n```\n\n")

        prompt_parts.append(
            "\nProvide ONLY additional findings not covered by security scan or duplicate detection.\n"
            "Format: [SEVERITY] File: <path> Line <num>: <description>\n"
            "Suggestion: <how to fix>\n"
        )

        return "".join(prompt_parts)

    def _parse_llm_response(self, response: str, contexts: List[AnalysisContext]) -> List[ReviewIssue]:
        """Parse LLM synthesis response into ReviewIssues."""
        issues = []

        # Map file paths from contexts for validation
        valid_files = {str(ctx.file_path) for ctx in contexts}

        current_issue = None
        current_suggestion_lines = []

        for line in response.strip().split("\n"):
            line = line.strip()

            # Check for severity markers
            for severity in IssueSeverity:
                marker = f"[{severity.value.upper()}]"
                if line.upper().startswith(marker):
                    # Save previous issue if exists
                    if current_issue:
                        if current_suggestion_lines:
                            current_issue.suggestion = " ".join(current_suggestion_lines)
                        issues.append(current_issue)

                    # Parse new issue
                    rest = line[len(marker):].strip()

                    # Try to extract file and line
                    file_path = None
                    line_num = None

                    # Look for "File: path" pattern
                    if "File:" in rest:
                        parts = rest.split("File:", 1)
                        if len(parts) > 1:
                            file_rest = parts[1].strip()
                            # Extract path until "Line" or ":"
                            for sep in ["Line", ":"]:
                                if sep in file_rest:
                                    file_path = file_rest.split(sep)[0].strip()
                                    break
                            else:
                                file_path = file_rest.split()[0] if file_rest else None

                    # Try to find line number
                    import re
                    line_match = re.search(r'[Ll]ine\s*(\d+)', rest)
                    if line_match:
                        line_num = int(line_match.group(1))

                    # Extract message (everything after file/line info)
                    message_start = rest.rfind(":") + 1 if ":" in rest else 0
                    message = rest[message_start:].strip() or rest

                    current_issue = ReviewIssue(
                        severity=severity,
                        file=Path(file_path) if file_path else contexts[0].file_path,
                        line=line_num,
                        message=message,
                    )
                    current_suggestion_lines = []
                    break
            else:
                # Not a severity line
                if line.lower().startswith("suggestion:"):
                    current_suggestion_lines.append(line[11:].strip())
                elif current_suggestion_lines and line:
                    current_suggestion_lines.append(line)

        # Don't forget the last issue
        if current_issue:
            if current_suggestion_lines:
                current_issue.suggestion = " ".join(current_suggestion_lines)
            issues.append(current_issue)

        return issues

    def _generate_summary(self, result: ReviewResult, contexts: List[AnalysisContext]) -> str:
        """Generate review summary."""
        parts = [
            f"{len(result.files_changed)} file{'s' if len(result.files_changed) != 1 else ''} changed",
            f"+{result.total_additions} -{result.total_deletions}",
        ]

        # Count findings by source
        security_count = sum(len(ctx.security_issues) for ctx in contexts)
        duplicate_count = sum(
            len([s for s in ctx.similar_code if s.similarity >= self.DUPLICATE_THRESHOLD])
            for ctx in contexts
        )

        if security_count:
            parts.append(f"{security_count} security")
        if duplicate_count:
            parts.append(f"{duplicate_count} duplicate{'s' if duplicate_count != 1 else ''}")
        if result.critical_count:
            parts.append(f"{result.critical_count} critical")
        if result.warning_count:
            parts.append(f"{result.warning_count} warning{'s' if result.warning_count != 1 else ''}")

        return " | ".join(parts)

    def _generate_pr_description(
        self,
        changes: List[FileChange],
        contexts: List[AnalysisContext]
    ) -> str:
        """Generate PR description based on analysis."""
        if not self.llm:
            return self._generate_simple_pr_description(changes)

        # Build context-aware prompt
        files_summary = "\n".join([
            f"- {c.path} ({c.status}: +{c.additions} -{c.deletions})"
            for c in changes[:20]
        ])

        analysis_summary = []
        for ctx in contexts[:10]:
            summary = f"- {ctx.file_path}"
            if ctx.security_issues:
                summary += f" ({len(ctx.security_issues)} security issues)"
            if ctx.similar_code:
                summary += f" (similar to existing code)"
            analysis_summary.append(summary)

        prompt = f"""Generate a concise PR description for these changes.

Files changed:
{files_summary}

Analysis summary:
{chr(10).join(analysis_summary)}

Total: +{sum(c.additions for c in changes)} -{sum(c.deletions for c in changes)}

Generate:
1. Brief summary (1-2 sentences)
2. Key changes (bullet points)
3. Any concerns from analysis
"""

        try:
            return self.llm.generate(
                prompt=prompt,
                system_prompt="Generate clear, concise PR descriptions. Be professional.",
                temperature=0.3,
            ).strip()
        except Exception:
            return self._generate_simple_pr_description(changes)

    def _generate_simple_pr_description(self, changes: List[FileChange]) -> str:
        """Generate simple PR description without LLM."""
        lines = [
            "## Changes",
            "",
            f"This PR includes changes to {len(changes)} files.",
            "",
            "### Files Modified",
        ]

        for change in changes[:10]:
            status_emoji = {"A": "✨", "M": "📝", "D": "🗑️", "R": "📦"}.get(change.status, "❓")
            lines.append(f"- {status_emoji} `{change.path}` (+{change.additions}, -{change.deletions})")

        if len(changes) > 10:
            lines.append(f"- ... and {len(changes) - 10} more files")

        return "\n".join(lines)


# Synthesis prompt for LLM (used only for additional insights)
SYNTHESIS_SYSTEM_PROMPT = """You are a senior code reviewer providing additional insights.

You are given analysis context that already includes:
- Security issues found by static analysis
- Duplicate/similar code detected by semantic search

Your job is to provide ADDITIONAL insights not covered by automated analysis:
- Logic errors or bugs
- Missing edge cases or error handling
- Performance concerns
- Best practice violations
- Code clarity issues

Be concise. Only report meaningful issues, not style nitpicks.
Format each issue as: [SEVERITY] File: <path> Line <num>: <description>
Followed by: Suggestion: <how to fix>

Severities: CRITICAL (blocks merge), WARNING (should fix), SUGGESTION (nice to have), PRAISE (good practice)
"""

"""
Validator for skill-distillation edits.

Two measurement axes:
  1. StructuralDistance  — Graph Edit Distance between two ExecutionStructure
                           path-graphs, labelled with (tier1, tier2) pairs.
  2. ResultQuality       — Whether the weak agent's output files are present and
                           have the same content as the strong agent's reference
                           outputs.  Comparison is type-aware: tabular (CSV/TSV),
                           JSON, plain-text, XLSX, and a byte-equality fallback
                           for everything else.  New file types can be added by
                           registering a FileComparator in COMPARATOR_REGISTRY.
"""

from __future__ import annotations

import abc
import csv
import hashlib
import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from execution_structure import ExecutionStructure


# ──────────────────────────────────────────────────────────────────────────────
# File content comparison
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
# FileCompareResult
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class FileCompareResult:
    filename: str               # strong agent's filename
    present_in_weak: bool
    # Which FileComparator handled this file (e.g. "tabular", "json", "text", "bytes")
    comparator: str = ""
    # Actual weak filename matched (may differ from filename when matched by content)
    matched_weak_filename: Optional[str] = None
    # None when file is absent from weak
    content_match: Optional[bool] = None
    # Continuous quality score in [0, 1].  Set by tabular comparators (cell-level
    # accuracy); None for binary comparators — those fall back to 1.0 / 0.0.
    content_score: Optional[float] = None
    # Type-specific comparison detail (populated by each comparator)
    detail: dict = field(default_factory=dict)
    notes: str = ""

    @property
    def name_mismatch(self) -> bool:
        """True when the weak file was matched by content, not by name."""
        return (
            self.matched_weak_filename is not None
            and self.matched_weak_filename != self.filename
        )

    def to_dict(self) -> dict:
        d: dict = {
            "filename": self.filename,
            "present_in_weak": self.present_in_weak,
            "comparator": self.comparator,
            "content_match": self.content_match,
        }
        if self.content_score is not None:
            d["content_score"] = round(self.content_score, 4)
        if self.matched_weak_filename and self.matched_weak_filename != self.filename:
            d["matched_weak_filename"] = self.matched_weak_filename
            d["name_mismatch"] = True
        if self.detail:
            d["detail"] = self.detail
        if self.notes:
            d["notes"] = self.notes
        return d


# ──────────────────────────────────────────────────────────────────────────────
# FileComparator — abstract base + concrete implementations
# ──────────────────────────────────────────────────────────────────────────────

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


class FileComparator(abc.ABC):
    """
    Strategy for comparing two output files of a specific type.

    To add support for a new file type:
      1. Subclass FileComparator and implement ``fingerprint`` + ``compare``.
      2. Register the instance in ``COMPARATOR_REGISTRY`` keyed by file extension
         (lower-case, with leading dot, e.g. ``".parquet"``).
    """

    @abc.abstractmethod
    def fingerprint(self, path: Path) -> str:
        """
        Return a stable content hash used for name-agnostic file matching.
        Must be order-independent for tabular formats (same data, different row
        order → same fingerprint).
        """

    @abc.abstractmethod
    def compare(
        self,
        strong: Path,
        weak: Path,
        weak_filename: str,
    ) -> FileCompareResult:
        """Compare strong and weak files; return a populated FileCompareResult."""


# ── Tabular (CSV / TSV) ───────────────────────────────────────────────────────

def _read_delimited(path: Path, delimiter: str) -> tuple[list[str], list[tuple[str, ...]]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    rows = [tuple(c.strip() for c in row) for row in reader if any(c.strip() for c in row)]
    if not rows:
        return [], []
    return list(rows[0]), rows[1:]


def _pair_rows(
    s_rows: list[tuple[str, ...]],
    w_rows: list[tuple[str, ...]],
) -> list[tuple[tuple[str, ...], "tuple[str, ...] | None"]]:
    """
    Pair each strong row with a weak row for cell-level comparison.

    Strategy:
      Pass 1 — consume exact (full-row) matches first.
      Pass 2 — for remaining unmatched strong rows, greedily pick the weak
               row with the highest cell-similarity score.

    Returns a list of (s_row, w_row_or_None) in strong-row order.
    """
    w_available = list(w_rows)
    pairs: list[tuple[tuple[str, ...], "tuple[str, ...] | None"]] = []

    # Pass 1: exact matches
    remaining_strong: list[tuple[str, ...]] = []
    for s_row in s_rows:
        found = False
        for i, w_row in enumerate(w_available):
            if w_row == s_row:
                pairs.append((s_row, w_row))
                w_available.pop(i)
                found = True
                break
        if not found:
            remaining_strong.append(s_row)

    # Pass 2: best partial matches
    for s_row in remaining_strong:
        if not w_available:
            pairs.append((s_row, None))
            continue
        ncols = max(len(s_row), max((len(w) for w in w_available), default=0))

        def _cell_sim(w_row: tuple[str, ...]) -> int:
            return sum(1 for a, b in zip(s_row, w_row) if a == b)

        best_idx = max(range(len(w_available)), key=lambda i: _cell_sim(w_available[i]))
        pairs.append((s_row, w_available.pop(best_idx)))

    return pairs


def _cell_level_metrics(
    s_rows: list[tuple[str, ...]],
    w_rows: list[tuple[str, ...]],
    headers: list[str] | None = None,
) -> dict:
    """
    Compute fine-grained tabular metrics using _pair_rows pairing.

    Returns a dict with:
      cell_match_ratio         — fraction of cells that match (all rows, all cols)
      per_column_match_ratio   — {col_name: ratio} if headers are provided
    """
    if not s_rows:
        return {"cell_match_ratio": 1.0}

    pairs = _pair_rows(s_rows, w_rows)
    ncols = len(s_rows[0]) if s_rows else 0

    total_cells = len(s_rows) * ncols if ncols > 0 else 0
    correct_cells = 0
    per_col_correct = [0] * ncols

    for s_row, w_row in pairs:
        if w_row is None:
            continue
        for j in range(min(len(s_row), len(w_row), ncols)):
            if s_row[j] == w_row[j]:
                correct_cells += 1
                per_col_correct[j] += 1

    cell_match_ratio = correct_cells / total_cells if total_cells > 0 else 1.0

    result: dict = {"cell_match_ratio": round(cell_match_ratio, 4)}
    if headers and ncols > 0:
        per_col: dict[str, float] = {}
        n_strong = len(s_rows)
        for j, col in enumerate(headers[:ncols]):
            per_col[col] = round(per_col_correct[j] / n_strong, 4) if n_strong > 0 else 1.0
        result["per_column_match_ratio"] = per_col

    return result


class TabularComparator(FileComparator):
    """Semantic comparison for delimiter-separated tabular files (CSV, TSV)."""

    def __init__(self, delimiter: str = ",") -> None:
        self._delim = delimiter

    def fingerprint(self, path: Path) -> str:
        try:
            _, rows = _read_delimited(path, self._delim)
            # Normalize whitespace within cells so that multi-line quoted cells
            # ("foo\nbar") match their single-line equivalents ("foo bar").
            normalized = [tuple(" ".join(c.split()) for c in row) for row in rows]
            return hashlib.sha256(repr(sorted(frozenset(normalized))).encode()).hexdigest()
        except Exception:
            return _sha256(path)

    @staticmethod
    def _normalize_rows(
        rows: list[tuple[str, ...]],
    ) -> list[tuple[str, ...]]:
        """Collapse internal whitespace/newlines in every cell (consistent with fingerprint)."""
        return [tuple(" ".join(c.split()) for c in row) for row in rows]

    def compare(self, strong: Path, weak: Path, weak_filename: str) -> FileCompareResult:
        try:
            s_headers, s_rows = _read_delimited(strong, self._delim)
            w_headers, w_rows = _read_delimited(weak,   self._delim)
        except Exception as exc:
            return FileCompareResult(
                filename=strong.name, present_in_weak=True,
                comparator="tabular", matched_weak_filename=weak_filename,
                content_match=False, notes=f"Parse error: {exc}",
            )

        # Normalize whitespace within cells (same logic as fingerprint) so that
        # multi-line quoted cells match their single-line equivalents.
        s_headers_n = [" ".join(h.split()) for h in s_headers]
        w_headers_n = [" ".join(h.split()) for h in w_headers]
        s_rows_n    = self._normalize_rows(s_rows)
        w_rows_n    = self._normalize_rows(w_rows)

        cols_match = s_headers_n == w_headers_n

        w_remaining = list(w_rows_n)
        matched = 0
        for row in s_rows_n:
            if row in w_remaining:
                w_remaining.remove(row)
                matched += 1

        total = len(s_rows_n)
        row_ratio = matched / total if total > 0 else 1.0
        missing = total - matched
        extra   = max(len(w_rows_n) - matched, 0)
        exact_match = cols_match and row_ratio == 1.0 and extra == 0

        # Fine-grained cell-level and per-column metrics
        cell_metrics = _cell_level_metrics(s_rows_n, w_rows_n, headers=s_headers_n)
        cell_ratio = cell_metrics["cell_match_ratio"]

        detail: dict = {
            "columns_match": cols_match,
            "row_match_ratio": round(row_ratio, 4),
            "cell_match_ratio": round(cell_ratio, 4),
            "strong_rows": total,
            "weak_rows": len(w_rows_n),
            "missing_rows": missing,
            "extra_rows": extra,
        }
        if "per_column_match_ratio" in cell_metrics:
            detail["per_column_match_ratio"] = cell_metrics["per_column_match_ratio"]

        return FileCompareResult(
            filename=strong.name, present_in_weak=True,
            comparator="tabular", matched_weak_filename=weak_filename,
            content_match=exact_match,
            content_score=cell_ratio if not exact_match else 1.0,
            detail=detail,
        )


# ── JSON ──────────────────────────────────────────────────────────────────────

class JsonComparator(FileComparator):
    """
    Structural comparison for JSON files.
    Normalises by sorting object keys and re-serialising before hashing/comparing.
    """

    @staticmethod
    def _normalise(path: Path) -> str:
        raw = path.read_text(encoding="utf-8", errors="replace")
        return json.dumps(json.loads(raw), sort_keys=True, ensure_ascii=False)

    def fingerprint(self, path: Path) -> str:
        try:
            normalised = self._normalise(path)
            return hashlib.sha256(normalised.encode()).hexdigest()
        except Exception:
            return _sha256(path)

    def compare(self, strong: Path, weak: Path, weak_filename: str) -> FileCompareResult:
        try:
            s_norm = self._normalise(strong)
            w_norm = self._normalise(weak)
        except Exception as exc:
            return FileCompareResult(
                filename=strong.name, present_in_weak=True,
                comparator="json", matched_weak_filename=weak_filename,
                content_match=False, notes=f"JSON parse error: {exc}",
            )

        match = s_norm == w_norm
        detail: dict = {"content_match": match}
        if not match:
            # Surface top-level key diff for quick inspection
            try:
                s_obj = json.loads(s_norm)
                w_obj = json.loads(w_norm)
                if isinstance(s_obj, dict) and isinstance(w_obj, dict):
                    s_keys = set(s_obj)
                    w_keys = set(w_obj)
                    detail["only_in_strong"] = sorted(s_keys - w_keys)
                    detail["only_in_weak"]   = sorted(w_keys - s_keys)
            except Exception:
                pass

        return FileCompareResult(
            filename=strong.name, present_in_weak=True,
            comparator="json", matched_weak_filename=weak_filename,
            content_match=match, detail=detail,
        )


# ── Plain text ────────────────────────────────────────────────────────────────

class TextComparator(FileComparator):
    """
    Line-level comparison for plain-text files (.txt, .md, .log, etc.).
    Strips trailing whitespace per line and ignores blank lines at EOF.
    """

    @staticmethod
    def _normalise(path: Path) -> list[str]:
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = [line.rstrip() for line in text.splitlines()]
        # Drop trailing blank lines
        while lines and not lines[-1]:
            lines.pop()
        return lines

    def fingerprint(self, path: Path) -> str:
        try:
            lines = self._normalise(path)
            return hashlib.sha256("\n".join(lines).encode()).hexdigest()
        except Exception:
            return _sha256(path)

    def compare(self, strong: Path, weak: Path, weak_filename: str) -> FileCompareResult:
        try:
            s_lines = self._normalise(strong)
            w_lines = self._normalise(weak)
        except Exception as exc:
            return FileCompareResult(
                filename=strong.name, present_in_weak=True,
                comparator="text", matched_weak_filename=weak_filename,
                content_match=False, notes=f"Read error: {exc}",
            )

        match = s_lines == w_lines
        return FileCompareResult(
            filename=strong.name, present_in_weak=True,
            comparator="text", matched_weak_filename=weak_filename,
            content_match=match,
            detail={
                "strong_lines": len(s_lines),
                "weak_lines": len(w_lines),
            },
        )


# ── XLSX ──────────────────────────────────────────────────────────────────────

class XlsxComparator(FileComparator):
    """
    Sheet-level comparison for Excel files via openpyxl.
    Compares sheet names and cell values (as strings); ignores formatting.
    Falls back to byte comparison if openpyxl is not installed.
    """

    @staticmethod
    def _sheets(path: Path) -> dict[str, list[tuple]]:
        import openpyxl  # noqa: PLC0415
        # Load twice: once for cached computed values, once for formula strings.
        # For a cell: prefer the computed value if present; fall back to the
        # formula string (e.g. "=SUMPRODUCT(...)") so that freshly-saved files
        # whose formulas have never been evaluated by Excel are still comparable.
        wb_vals = openpyxl.load_workbook(path, read_only=True, data_only=True)
        wb_form = openpyxl.load_workbook(path, read_only=True, data_only=False)
        result: dict[str, list[tuple]] = {}
        for name in wb_vals.sheetnames:
            ws_v = wb_vals[name]
            ws_f = wb_form[name]
            rows_v = list(ws_v.iter_rows())
            rows_f = list(ws_f.iter_rows())
            sheet_rows = []
            for rv, rf in zip(rows_v, rows_f):
                has_content = False
                cells = []
                for cv, cf in zip(rv, rf):
                    if cv.value is not None:
                        cells.append(str(cv.value).strip())
                        has_content = True
                    elif cf.value is not None:
                        cells.append(str(cf.value).strip())
                        has_content = True
                    else:
                        cells.append("")
                if has_content:
                    sheet_rows.append(tuple(cells))
            result[name] = sheet_rows
        wb_vals.close()
        wb_form.close()
        return result

    def fingerprint(self, path: Path) -> str:
        try:
            sheets = self._sheets(path)
            canon = json.dumps(sheets, sort_keys=True, default=str)
            return hashlib.sha256(canon.encode()).hexdigest()
        except Exception:
            return _sha256(path)

    def compare(self, strong: Path, weak: Path, weak_filename: str) -> FileCompareResult:
        try:
            s_sheets = self._sheets(strong)
            w_sheets = self._sheets(weak)
        except ImportError:
            # openpyxl not installed — fall back to byte equality
            s_hash, w_hash = _sha256(strong), _sha256(weak)
            return FileCompareResult(
                filename=strong.name, present_in_weak=True,
                comparator="xlsx_bytes_fallback", matched_weak_filename=weak_filename,
                content_match=s_hash == w_hash,
                detail={"strong_sha256": s_hash, "weak_sha256": w_hash},
                notes="openpyxl not installed; used byte comparison",
            )
        except Exception as exc:
            return FileCompareResult(
                filename=strong.name, present_in_weak=True,
                comparator="xlsx", matched_weak_filename=weak_filename,
                content_match=False, notes=f"XLSX parse error: {exc}",
            )

        s_names = list(s_sheets)
        w_names = list(w_sheets)
        sheets_match = s_names == w_names

        per_sheet_detail: dict[str, dict] = {}
        sheet_cell_ratios: list[float] = []

        for sname in s_names:
            if sname not in w_sheets:
                per_sheet_detail[sname] = {
                    "present": False,
                    "row_match_ratio": 0.0,
                    "cell_match_ratio": 0.0,
                }
                sheet_cell_ratios.append(0.0)
                continue

            s_rows = s_sheets[sname]
            w_rows = w_sheets[sname]

            # Exact row matching (order-independent)
            w_remaining = list(w_rows)
            matched = 0
            for row in s_rows:
                if row in w_remaining:
                    w_remaining.remove(row)
                    matched += 1

            total = len(s_rows)
            row_ratio = matched / total if total > 0 else 1.0
            extra = max(len(w_rows) - matched, 0)
            sheet_exact = row_ratio == 1.0 and extra == 0

            # Cell-level metrics (infer headers from first row if any)
            if s_rows:
                headers = [str(c) for c in s_rows[0]] if s_rows else None
                data_s = s_rows[1:] if len(s_rows) > 1 else s_rows
                data_w = w_rows[1:] if len(w_rows) > 1 else w_rows
                cell_metrics = _cell_level_metrics(data_s, data_w, headers=headers)
            else:
                cell_metrics = {"cell_match_ratio": 1.0}

            cell_ratio = cell_metrics["cell_match_ratio"]
            sheet_cell_ratios.append(cell_ratio if not sheet_exact else 1.0)

            sheet_info: dict = {
                "present": True,
                "row_match_ratio": round(row_ratio, 4),
                "cell_match_ratio": round(cell_ratio, 4),
                "strong_rows": total,
                "weak_rows": len(w_rows),
                "missing_rows": total - matched,
                "extra_rows": extra,
            }
            if "per_column_match_ratio" in cell_metrics:
                sheet_info["per_column_match_ratio"] = cell_metrics["per_column_match_ratio"]
            per_sheet_detail[sname] = sheet_info

        all_match = sheets_match and all(
            d.get("row_match_ratio", 0.0) == 1.0 and d.get("extra_rows", 1) == 0
            for d in per_sheet_detail.values()
        )
        overall_cell_ratio = (
            sum(sheet_cell_ratios) / len(sheet_cell_ratios)
            if sheet_cell_ratios else 1.0
        )

        return FileCompareResult(
            filename=strong.name, present_in_weak=True,
            comparator="xlsx", matched_weak_filename=weak_filename,
            content_match=all_match,
            content_score=overall_cell_ratio if not all_match else 1.0,
            detail={
                "sheets_match": sheets_match,
                "strong_sheets": s_names,
                "weak_sheets": w_names,
                "cell_match_ratio": round(overall_cell_ratio, 4),
                "per_sheet": per_sheet_detail,
            },
        )


# ── Bytes fallback ────────────────────────────────────────────────────────────

class BytesComparator(FileComparator):
    """SHA256 byte-equality fallback for any unrecognised file type."""

    def fingerprint(self, path: Path) -> str:
        return _sha256(path)

    def compare(self, strong: Path, weak: Path, weak_filename: str) -> FileCompareResult:
        s_hash = _sha256(strong)
        w_hash = _sha256(weak)
        return FileCompareResult(
            filename=strong.name, present_in_weak=True,
            comparator="bytes", matched_weak_filename=weak_filename,
            content_match=s_hash == w_hash,
            detail={"strong_sha256": s_hash, "weak_sha256": w_hash},
        )


# ──────────────────────────────────────────────────────────────────────────────
# Comparator registry
# ──────────────────────────────────────────────────────────────────────────────

#: Map file extension (lower-case, with dot) → FileComparator instance.
#: Add entries here to support additional file types.
COMPARATOR_REGISTRY: dict[str, FileComparator] = {
    ".csv":  TabularComparator(","),
    ".tsv":  TabularComparator("\t"),
    ".json": JsonComparator(),
    ".txt":  TextComparator(),
    ".md":   TextComparator(),
    ".log":  TextComparator(),
    ".xlsx": XlsxComparator(),
    ".xls":  XlsxComparator(),
}

_BYTES_COMPARATOR = BytesComparator()


def get_comparator(path: Path) -> FileComparator:
    """Return the appropriate FileComparator for *path* based on its extension."""
    return COMPARATOR_REGISTRY.get(path.suffix.lower(), _BYTES_COMPARATOR)


# ──────────────────────────────────────────────────────────────────────────────
# Public comparison entry-point + name-agnostic matching helpers
# ──────────────────────────────────────────────────────────────────────────────

def compare_file(
    strong_path: Path,
    weak_path: Path,
    *,
    weak_filename: str | None = None,
) -> FileCompareResult:
    """
    Compare the content of one output file from strong and weak agents.
    Dispatches to the registered FileComparator for the file's extension.

    ``weak_filename`` is the actual name of the matched weak file.  When it
    differs from ``strong_path.name`` (content-based match), it is recorded
    in ``FileCompareResult.matched_weak_filename``.
    """
    if not weak_path.exists():
        return FileCompareResult(filename=strong_path.name, present_in_weak=False)

    comparator = get_comparator(strong_path)
    return comparator.compare(strong_path, weak_path, weak_filename or weak_path.name)


def _file_fingerprint(path: Path) -> str:
    """Content fingerprint via the registered comparator (for name-agnostic matching)."""
    return get_comparator(path).fingerprint(path)


def _match_files_by_content(
    unmatched_strong: dict[str, Path],
    unmatched_weak: dict[str, Path],
) -> list[tuple[str, str]]:
    """
    Greedily match strong files to weak files by content fingerprint.

    Returns a list of (strong_name, weak_name) pairs where fingerprints match.
    Only exact fingerprint matches are paired.
    """
    if not unmatched_strong or not unmatched_weak:
        return []

    weak_by_fp: dict[str, list[str]] = {}
    for name, path in unmatched_weak.items():
        fp = _file_fingerprint(path)
        weak_by_fp.setdefault(fp, []).append(name)

    pairs: list[tuple[str, str]] = []
    used_weak: set[str] = set()

    for s_name, s_path in unmatched_strong.items():
        fp = _file_fingerprint(s_path)
        candidates = [n for n in weak_by_fp.get(fp, []) if n not in used_weak]
        if candidates:
            w_name = candidates[0]
            pairs.append((s_name, w_name))
            used_weak.add(w_name)

    return pairs


# ──────────────────────────────────────────────────────────────────────────────
# Result quality report
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ResultQuality:
    """Comparison of output files between strong and weak agent runs."""

    file_results: list[FileCompareResult] = field(default_factory=list)

    # Files in strong output but absent from weak
    missing_files: list[str] = field(default_factory=list)
    # Files in weak output but not in strong (unexpected extras)
    extra_files: list[str] = field(default_factory=list)

    @property
    def file_presence_ratio(self) -> float:
        """Fraction of strong output files that are present in weak output."""
        total = len(self.file_results) + len(self.missing_files)
        if total == 0:
            return 1.0
        return len(self.file_results) / total

    @property
    def content_match_ratio(self) -> float:
        """
        Average content quality score across present files.

        For tabular files (CSV, XLSX) this is the cell-level match ratio —
        a continuous value in [0, 1] that reflects partial correctness.
        For all other file types it falls back to binary 1.0 / 0.0.
        """
        if not self.file_results:
            return 0.0
        scores = [
            r.content_score if r.content_score is not None
            else (1.0 if r.content_match else 0.0)
            for r in self.file_results
        ]
        return sum(scores) / len(scores)

    @property
    def overall_quality_score(self) -> float:
        """
        Composite quality score in [0, 1].

        Weights:
          - file presence ratio : 0.4
          - content match ratio : 0.6
        """
        return (
            0.4 * self.file_presence_ratio
            + 0.6 * self.content_match_ratio
        )

    def to_dict(self) -> dict:
        return {
            "files": {
                "present": [r.to_dict() for r in self.file_results],
                "missing_from_weak": self.missing_files,
                "extra_in_weak": self.extra_files,
            },
            "scores": {
                "file_presence_ratio": round(self.file_presence_ratio, 4),
                "content_match_ratio": round(self.content_match_ratio, 4),
                "overall_quality_score": round(self.overall_quality_score, 4),
            },
        }


# ──────────────────────────────────────────────────────────────────────────────
# Validation report
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ValidationReport:
    """Result of the Validator: result quality only.
    Structural distance is computed by the GapAgent pipeline."""

    strong_agent: str
    weak_agent: str
    result_quality: ResultQuality

    @property
    def verdict(self) -> str:
        """
        Verdict logic (file content only):

          PASS    — all expected files present, full content match, no extra files.
          PARTIAL — quality score ≥ 0.5.
          FAIL    — quality score < 0.5.
        """
        rq = self.result_quality
        if (
            rq.file_presence_ratio == 1.0
            and rq.content_match_ratio == 1.0
            and not rq.extra_files
        ):
            return "PASS"
        if rq.overall_quality_score >= 0.5:
            return "PARTIAL"
        return "FAIL"

    def summary_text(self) -> str:
        rq = self.result_quality
        lines = [
            f"Validation: {self.weak_agent}  vs  {self.strong_agent}",
            f"  Verdict             : {self.verdict}",
            f"  Result quality score: {rq.overall_quality_score:.3f}",
            f"    File presence     : {rq.file_presence_ratio:.3f}",
            f"    Content match     : {rq.content_match_ratio:.3f}",
        ]
        if rq.missing_files:
            lines.append(f"    Missing files     : {rq.missing_files}")
        for fc in rq.file_results:
            status = "✓" if fc.content_match else "✗"
            name = fc.filename
            if fc.name_mismatch:
                name += f" → {fc.matched_weak_filename}"
            d = fc.detail
            score_str = (
                f"  score={fc.content_score:.3f}" if fc.content_score is not None and not fc.content_match
                else ""
            )
            if fc.comparator == "tabular":
                extra = (
                    f"{score_str}"
                    f"  rows={d.get('row_match_ratio', 0.0):.3f}"
                    f"  cells={d.get('cell_match_ratio', 0.0):.3f}"
                    f"  missing={d.get('missing_rows', 0)}"
                    f"  extra={d.get('extra_rows', 0)}"
                )
            elif fc.comparator == "xlsx":
                extra = (
                    f"{score_str}"
                    f"  cells={d.get('cell_match_ratio', 0.0):.3f}"
                    f"  sheets={d.get('strong_sheets', [])}"
                )
            else:
                extra = f"  [{fc.comparator}]{score_str}"
            lines.append(f"    {status} {name}{extra}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "strong_agent": self.strong_agent,
            "weak_agent": self.weak_agent,
            "verdict": self.verdict,
            "result_quality": self.result_quality.to_dict(),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Validator
# ──────────────────────────────────────────────────────────────────────────────

class Validator:
    """
    Validates a skill-distillation edit by comparing output file content.
    Structural analysis (GED) is the responsibility of the GapAgent pipeline.
    """

    def validate(
        self,
        strong_structure: ExecutionStructure,
        weak_structure: ExecutionStructure,
        *,
        strong_results_dir: Path | None = None,
        weak_results_dir: Path | None = None,
        output_path: Path | None = None,
    ) -> ValidationReport:
        """
        Parameters
        ----------
        strong_structure   ExecutionStructure for the strong (reference) agent.
        weak_structure     ExecutionStructure for the weak (candidate) agent.
        strong_results_dir Directory containing output files from the strong run.
                           Optional; if omitted, ResultQuality covers outcome only.
        weak_results_dir   Directory containing output files from the weak run.
        output_path        If set, write the ValidationReport JSON here.
        """
        rq = self._result_quality(
            strong_structure,
            weak_structure,
            strong_results_dir=strong_results_dir,
            weak_results_dir=weak_results_dir,
        )
        report = ValidationReport(
            strong_agent=strong_structure.agent_id,
            weak_agent=weak_structure.agent_id,
            result_quality=rq,
        )
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(report.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        return report

    def _result_quality(
        self,
        strong: ExecutionStructure,
        weak: ExecutionStructure,
        *,
        strong_results_dir: Path | None,
        weak_results_dir: Path | None,
    ) -> ResultQuality:
        rq = ResultQuality()

        # No result directories → outcome-only quality
        if strong_results_dir is None or weak_results_dir is None:
            return rq

        # Suffixes that are always intermediate execution artifacts, never final outputs.
        # Applied to both sides so scripts/logs never pollute the comparison.
        _INTERMEDIATE_SUFFIXES = {".py", ".pyc", ".sh", ".bash", ".log", ".tmp"}

        def _is_output_file(p: Path) -> bool:
            return p.is_file() and p.suffix.lower() not in _INTERMEDIATE_SUFFIXES

        # Enumerate output files on disk (skip intermediate artifacts)
        strong_files = {p.name: p for p in strong_results_dir.iterdir() if _is_output_file(p)}
        weak_files   = {p.name: p for p in weak_results_dir.iterdir()   if _is_output_file(p)}

        # Cross-check: strong produced_files list (if populated) guides which files matter.
        # Filter it by suffix too, in case the abstractor captured intermediate scripts
        # alongside the real outputs.
        expected_names: list[str]
        if strong.produced_files:
            expected_names = [
                name for name in strong.produced_files
                if Path(name).suffix.lower() not in _INTERMEDIATE_SUFFIXES
            ]
        else:
            expected_names = sorted(strong_files.keys())

        # ── Content-based matching (names are irrelevant) ────────────────────
        # Each strong file is paired with the weak file whose content fingerprint
        # matches.  Filenames are never compared — only content matters.
        all_strong: dict[str, Path] = {}
        for name in expected_names:
            s_path = strong_files.get(name)
            if s_path is not None:
                all_strong[name] = s_path

        content_pairs = _match_files_by_content(all_strong, dict(weak_files))
        content_matched_strong = {s for s, _ in content_pairs}
        matched_weak_names: set[str] = set()

        for s_name, w_name in content_pairs:
            s_path = all_strong[s_name]
            w_path = weak_files[w_name]
            rq.file_results.append(
                compare_file(s_path, w_path, weak_filename=w_name)
            )
            matched_weak_names.add(w_name)

        # Strong files whose content was not found anywhere in weak → missing
        for name in all_strong:
            if name not in content_matched_strong:
                rq.missing_files.append(name)

        # Weak files not matched to any strong file → extra
        for name in weak_files:
            if name not in matched_weak_names:
                rq.extra_files.append(name)

        return rq

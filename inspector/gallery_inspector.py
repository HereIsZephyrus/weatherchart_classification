"""
Gallery Inspector class

This module provides a utility to inspect the downloaded gallery dataset,
computing counts per kind, per projection, date coverage, and missing items.

Expected directory layout (after downloads):
  gallery/
    └── <kind>/
         └── <YYYYMMDDHHMM>_<projection>.webp
"""

from __future__ import annotations

import os
import re
import logging
from dataclasses import dataclass, field
from html import escape
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Accept broader filename patterns from historical downloads:
# 1) <date>_<projection>.webp (canonical)
# 2) <prefix>_<date>_<projection>.webp (prefix like medium-uv-z)
# 3) <prefix>_<date>.webp (projection omitted)
# In case projection is omitted, we record it as "unknown".
FILENAME_PATTERN = re.compile(
    r"^(?:.+?_)?(?P<date>\d{12})(?:_(?P<projection>[^.]+))?\.webp$",
    re.IGNORECASE,
)

DEFAULT_PROJECTIONS = [
    "opencharts_eastern_asia",
    "opencharts_eruasia",
    "opencharts_south_east_asia_and_indonesia",
    "opencharts_southern_asia",
]


@dataclass
class KindStats:
    """
    Construct statistics for a kind of images
    """
    kind_name: str
    image_count: int = 0
    per_projection_count: Dict[str, int] = field(default_factory=dict)
    unique_dates: Set[str] = field(default_factory=set)
    date_to_projections: Dict[str, Set[str]] = field(default_factory=dict)

    def register(self, date_str: str, projection: str) -> None:
        """
        Register an image
        """
        self.image_count += 1
        self.unique_dates.add(date_str)
        self.per_projection_count[projection] = self.per_projection_count.get(projection, 0) + 1
        if date_str not in self.date_to_projections:
            self.date_to_projections[date_str] = set()
        self.date_to_projections[date_str].add(projection)

    def date_range(self) -> Tuple[Optional[str], Optional[str]]:
        """
        Get the date range of the images
        """
        if not self.unique_dates:
            return None, None
        # Dates follow YYYYMMDDHHMM, lexical order equals chronological order
        dates_sorted = sorted(self.unique_dates)
        return dates_sorted[0], dates_sorted[-1]

    def missing_by_date(self, expected_projections: List[str]) -> Dict[str, List[str]]:
        """
        Get the missing projections by date
        """
        missing: Dict[str, List[str]] = {}
        expected_set = set(expected_projections)
        for date_str, present in self.date_to_projections.items():
            absent = list(expected_set - present)
            if absent:
                # Keep deterministic order similar to expected list
                missing[date_str] = [p for p in expected_projections if p in absent]
        return missing


@dataclass
class GalleryStats:
    """
    Construct statistics for the gallery
    """
    base_dir: str
    total_images: int = 0
    kinds: Dict[str, KindStats] = field(default_factory=dict)
    per_projection_total: Dict[str, int] = field(default_factory=dict)
    invalid_files: List[str] = field(default_factory=list)

    def register(self, kind: str, date_str: str, projection: str) -> None:
        """
        Register an image
        """
        self.total_images += 1
        if kind not in self.kinds:
            self.kinds[kind] = KindStats(kind_name=kind)
        self.kinds[kind].register(date_str, projection)
        self.per_projection_total[projection] = self.per_projection_total.get(projection, 0) + 1

    def global_date_range(self) -> Tuple[Optional[str], Optional[str]]:
        """
        Get the global date range of the images
        """
        all_dates: List[str] = []
        for ks in self.kinds.values():
            all_dates.extend(list(ks.unique_dates))
        if not all_dates:
            return None, None
        all_dates.sort()
        return all_dates[0], all_dates[-1]


class GalleryInspector:
    """
    Inspect a downloaded gallery dataset and summarize its information.

    - Counts per kind
    - Counts per projection (global and per kind)
    - Date coverage and range
    - Missing projections per date (by kind)
    """

    def __init__(self, base_dir: str, projections: Optional[List[str]] = None):
        self.base_dir = os.path.abspath(base_dir)
        self.expected_projections = projections[:] if projections else DEFAULT_PROJECTIONS[:]
        self.stats: Optional[GalleryStats] = None

    def inspect(self) -> GalleryStats:
        """
        Scan the `base_dir` and compute gallery statistics.
        Returns the computed GalleryStats.
        """
        stats = GalleryStats(base_dir=self.base_dir)

        if not os.path.isdir(self.base_dir):
            logger.warning("Gallery base directory does not exist: %s", self.base_dir)
            self.stats = stats
            return stats

        # Each subdirectory under base_dir is treated as a kind
        for entry in os.scandir(self.base_dir):
            if not entry.is_dir():
                continue
            kind_name = entry.name
            kind_path = entry.path

            for file_entry in os.scandir(kind_path):
                if not file_entry.is_file():
                    continue
                filename = file_entry.name
                if not filename.lower().endswith(".webp") and not filename.lower().endswith(".png"):
                    continue
                match = FILENAME_PATTERN.match(filename)
                if not match:
                    stats.invalid_files.append(os.path.join(kind_path, filename))
                    continue
                date_str = match.group("date")
                projection = match.group("projection") or "unknown"
                stats.register(kind_name, date_str, projection)

        self.stats = stats
        self._log_quick_summary(stats)
        return stats

    def _log_quick_summary(self, stats: GalleryStats) -> None:
        logger.info("Scanned gallery at %s", stats.base_dir)
        logger.info("Total images: %d | kinds: %d", stats.total_images, len(stats.kinds))
        if stats.per_projection_total:
            projections_line = ", ".join(
                f"{proj}: {cnt}" for proj, cnt in sorted(stats.per_projection_total.items())
            )
            logger.info("Per-projection total: %s", projections_line)
        gmin, gmax = stats.global_date_range()
        if gmin and gmax:
            logger.info("Global date range: %s ~ %s", gmin, gmax)
        if stats.invalid_files:
            logger.warning("Invalid filenames (pattern mismatch): %d", len(stats.invalid_files))

    def info(self, examples_per_kind: int = 3) -> None:
        """
        Print a concise human-readable summary to the logger.
        `examples_per_kind` limits how many missing-date examples to log per kind.
        """
        if self.stats is None:
            logger.info("No stats available. Call inspect() first.")
            return

        stats = self.stats
        for kind, kind_stats in sorted(stats.kinds.items()):
            kmin, kmax = kind_stats.date_range()
            logger.info(
                "Kind=%s | images=%d | unique_dates=%d | date_range=%s~%s",
                kind,
                kind_stats.image_count,
                len(kind_stats.unique_dates),
                kmin or "-",
                kmax or "-",
            )
            if kind_stats.per_projection_count:
                per_proj_line = ", ".join(
                    f"{proj}: {cnt}" for proj, cnt in sorted(kind_stats.per_projection_count.items())
                )
                logger.info("  per-projection: %s", per_proj_line)

            missing = kind_stats.missing_by_date(self.expected_projections)
            if missing:
                logger.info(
                    "  missing coverage: %d dates (expecting %d projections)",
                    len(missing),
                    len(self.expected_projections),
                )
                # Show a few representative dates
                for idx, (date_str, absent) in enumerate(sorted(missing.items())):
                    if idx >= examples_per_kind:
                        logger.info("  ... (and %d more dates)", len(missing) - examples_per_kind)
                        break
                    logger.info("    %s -> missing: %s", date_str, ", ".join(absent))
            else:
                logger.info("  full coverage for all listed projections")

        if stats.invalid_files:
            # Log a few invalid filenames for diagnostics
            preview = stats.invalid_files[:min(5, len(stats.invalid_files))]
            logger.warning("Invalid filename examples: %s", "; ".join(preview))

    def to_html(self, examples_per_kind: int = 3) -> str:
        """
        Convert the stats to an HTML report
        """
        if self.stats is None:
            return "<p>No stats available. Call inspect() first.</p>"

        stats = self.stats
        def html_escape(s: str) -> str:
            return escape(str(s))

        parts = []
        parts.append("<html><head><meta charset='utf-8'><title>Gallery Report</title>")
        parts.append("<style>body{font-family:Arial,Helvetica,sans-serif;}")
        parts.append("h1,h2,h3{margin:0.6em 0;} table{border-collapse:collapse;width:100%;margin:10px 0;}")
        parts.append("th,td{border:1px solid #ddd;padding:6px;text-align:left;}")
        parts.append("tr:nth-child(even){background:#fafafa;} .muted{color:#777;}")
        parts.append(".mono{font-family:monospace;}")
        parts.append("</style></head><body>")

        parts.append("<h1>ECMWF Gallery 数据集报告</h1>")
        parts.append(f"<p class='mono'>路径: {html_escape(stats.base_dir)}</p>")

        # Global summary
        parts.append("<h2>全局概览</h2>")
        parts.append("<table>")
        gmin, gmax = stats.global_date_range()
        parts.append("<tr><th>总图像数</th><td>" + str(stats.total_images) + "</td></tr>")
        parts.append("<tr><th>类别数</th><td>" + str(len(stats.kinds)) + "</td></tr>")
        parts.append("<tr><th>全局日期范围</th><td>" + html_escape(f"{gmin or '-'} ~ {gmax or '-'}") + "</td></tr>")
        if stats.per_projection_total:
            proj_line = ", ".join(f"{html_escape(p)}: {c}" for p, c in sorted(stats.per_projection_total.items()))
            parts.append("<tr><th>按投影计数</th><td>" + proj_line + "</td></tr>")
        parts.append("</table>")

        # Invalid filenames
        if stats.invalid_files:
            parts.append("<h3>无效文件名示例</h3><ul>")
            for path in stats.invalid_files[: min(10, len(stats.invalid_files))]:
                parts.append("<li class='mono'>" + html_escape(path) + "</li>")
            parts.append("</ul>")

        # Per-kind details
        parts.append("<h2>按类别明细</h2>")
        for kind, kind_stats in sorted(stats.kinds.items()):
            parts.append(f"<h3>{html_escape(kind)}</h3>")
            kmin, kmax = kind_stats.date_range()
            parts.append("<table>")
            parts.append("<tr><th>图像数</th><td>" + str(kind_stats.image_count) + "</td></tr>")
            parts.append("<tr><th>唯一日期数</th><td>" + str(len(kind_stats.unique_dates)) + "</td></tr>")
            parts.append("<tr><th>日期范围</th><td>" + html_escape(f"{kmin or '-'} ~ {kmax or '-'}") + "</td></tr>")
            if kind_stats.per_projection_count:
                proj = ", ".join(
                    f"{html_escape(p)}: {c}" for p, c in sorted(kind_stats.per_projection_count.items())
                )
                parts.append("<tr><th>按投影计数</th><td>" + proj + "</td></tr>")
            parts.append("</table>")

            missing = kind_stats.missing_by_date(self.expected_projections)
            if missing:
                parts.append(
                    f"<p>缺失覆盖：{len(missing)} 个日期（期望 {len(self.expected_projections)} 个投影）</p>"
                )
                parts.append("<table><tr><th>日期</th><th>缺失投影</th></tr>")
                for idx, (date_str, absent) in enumerate(sorted(missing.items())):
                    if idx >= examples_per_kind:
                        parts.append(
                            f"<tr><td colspan='2' class='muted'>... 以及 {len(missing) - examples_per_kind} 更多日期</td></tr>"
                        )
                        break
                    parts.append(
                        "<tr><td>" + html_escape(date_str) + "</td><td>" + html_escape(", ".join(absent)) + "</td></tr>"
                    )
                parts.append("</table>")
            else:
                parts.append("<p>所有列出的投影均完整覆盖。</p>")

        parts.append("</body></html>")
        return "".join(parts)

    def save_html(self, filepath: str, examples_per_kind: int = 3) -> None:
        """
        Save the HTML report to a file
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.to_html(examples_per_kind=examples_per_kind))

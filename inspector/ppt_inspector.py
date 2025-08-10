"""
PPT Inspector module

This module provides a utility to inspect the PPTX files in the `income/slides` directory,
and the extracted images in the `income/extracted_images` directory.

It computes counts per source, per extension, per date range, and per PPT.

Expected directory layout (after extraction):
  income/
    └── slides/
         └── <YYYYMMDD>-早会商-信息中心-实况.pptx
         └── aoc<YYYYMMDD>.pptx
         └── <YYYY年MM月DD日早间会商首席发言>.pptx
    └── extracted_images/
         └── <ppt_name>_slide<slide_index>_img<img_index>.<ext>

Naming conventions:
- PPT files:
  - CMA: YYYYMMDD-早会商-信息中心-实况.pptx
  - AOC: aocYYYYMMDD.pptx
  - NMC: YYYY年MM月DD日早间会商首席发言.pptx
- Image files: <ppt_name>_slide<slide_index>_img<img_index>.<ext>
"""

from __future__ import annotations

import os
import re
import logging
from dataclasses import dataclass, field
from html import escape
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# PPT name to source matching
PPT_PATTERNS = {
    "cma": re.compile(r"^(?P<date>\d{8})-早会商-信息中心-实况$"),
    "aoc": re.compile(r"^aoc(?P<date>\d{8})$"),
    "nmc": re.compile(r"^(?P<year>\d{4})年(?P<month>\d{2})月(?P<day>\d{2})日早间会商首席发言$"),
}

# Image file naming
IMAGE_PATTERN = re.compile(
    r"^(?P<ppt_name>.+?)_slide(?P<slide>\d+)_img(?P<img>\d+)\.(?P<ext>[a-zA-Z0-9]+)$"
)

def _safe_listdir(path: str) -> List[str]:
    try:
        return os.listdir(path)
    except FileNotFoundError:
        return []


@dataclass
class SlidesStats:
    """
    Statistics for PPTX files
    """
    base_dir: str
    total_pptx: int = 0
    by_source_count: Dict[str, int] = field(default_factory=dict)  # cma/aoc/nmc/unknown
    date_set: Set[str] = field(default_factory=set)  # 仅记录可解析出的 YYYYMMDD
    invalid_files: List[str] = field(default_factory=list)

    def register(self, ppt_name: str) -> None:
        """
        Register a PPTX file
        """
        source, yyyymmdd = classify_ppt_name(ppt_name)
        self.total_pptx += 1
        self.by_source_count[source] = self.by_source_count.get(source, 0) + 1
        if yyyymmdd:
            self.date_set.add(yyyymmdd)

    def date_range(self) -> Tuple[Optional[str], Optional[str]]:
        """
        Get the date range of the PPTX files
        """
        if not self.date_set:
            return None, None
        ordered = sorted(self.date_set)
        return ordered[0], ordered[-1]


@dataclass
class ImagesStats:
    """
    Statistics for extracted images
    """
    base_dir: str
    total_images: int = 0
    by_ext: Dict[str, int] = field(default_factory=dict)
    by_source_from_pptname: Dict[str, int] = field(default_factory=dict)
    ppt_to_images: Dict[str, int] = field(default_factory=dict)
    invalid_files: List[str] = field(default_factory=list)

    def register(self, filename: str) -> None:
        """
        Register an extracted image
        """
        self.total_images += 1
        m = IMAGE_PATTERN.match(filename)
        if not m:
            self.invalid_files.append(filename)
            return
        ppt_name = m.group("ppt_name")
        ext = m.group("ext").lower()
        self.by_ext[ext] = self.by_ext.get(ext, 0) + 1
        self.ppt_to_images[ppt_name] = self.ppt_to_images.get(ppt_name, 0) + 1
        source, _ = classify_ppt_name(ppt_name)
        self.by_source_from_pptname[source] = self.by_source_from_pptname.get(source, 0) + 1


def classify_ppt_name(ppt_name: str) -> Tuple[str, Optional[str]]:
    """
    Classify a PPT name into source and YYYYMMDD
    For nmc (Chinese date), return YYYYMMDD (zero-padded)
    """
    for source, pat in PPT_PATTERNS.items():
        m = pat.match(ppt_name)
        if not m:
            continue
        if source in ("cma", "aoc"):
            return source, m.group("date")
        # nmc: 中文日期
        year = m.group("year")
        month = m.group("month")
        day = m.group("day")
        return "nmc", f"{year}{month}{day}"
    return "unknown", None


class PPTInspector:
    """
    Inspect the `income` directory:
    - slides: statistics for PPTX files (source distribution, date range, invalid filenames)
    - extracted_images: statistics for extracted images (source distribution, extension distribution, invalid filenames, per-PPT counts)
    """

    def __init__(self, income_dir: str = "income"):
        self.income_dir = os.path.abspath(income_dir)
        self.slides_dir = os.path.join(self.income_dir, "slides")
        self.images_dir = os.path.join(self.income_dir, "extracted_images")
        self.slides_stats: Optional[SlidesStats] = None
        self.images_stats: Optional[ImagesStats] = None

    def inspect_slides(self) -> SlidesStats:
        """
        Inspect the slides directory
        """
        stats = SlidesStats(base_dir=self.slides_dir)
        if not os.path.isdir(self.slides_dir):
            logger.warning("Slides directory does not exist: %s", self.slides_dir)
            self.slides_stats = stats
            return stats
        for filename in _safe_listdir(self.slides_dir):
            if not filename.lower().endswith(".pptx"):
                continue
            ppt_name = os.path.splitext(filename)[0]
            # 记录是否命名合法
            src, _ = classify_ppt_name(ppt_name)
            if src == "unknown":
                stats.invalid_files.append(os.path.join(self.slides_dir, filename))
            stats.register(ppt_name)
        self.slides_stats = stats
        self._log_slides_summary(stats)
        return stats

    def inspect_images(self) -> ImagesStats:
        """
        Inspect the images directory
        """
        stats = ImagesStats(base_dir=self.images_dir)
        if not os.path.isdir(self.images_dir):
            logger.warning("Images directory does not exist: %s", self.images_dir)
            self.images_stats = stats
            return stats
        for filename in _safe_listdir(self.images_dir):
            if filename.startswith("."):
                continue
            stats.register(filename)
        self.images_stats = stats
        self._log_images_summary(stats)
        return stats

    def inspect_all(self) -> Tuple[SlidesStats, ImagesStats]:
        """
        Inspect all the directories
        """
        return self.inspect_slides(), self.inspect_images()

    def _log_slides_summary(self, stats: SlidesStats) -> None:
        logger.info("Scanned slides at %s", stats.base_dir)
        logger.info("Total pptx: %d", stats.total_pptx)
        if stats.by_source_count:
            items = ", ".join(f"{k}: {v}" for k, v in sorted(stats.by_source_count.items()))
            logger.info("By source: %s", items)
        dmin, dmax = stats.date_range()
        if dmin and dmax:
            logger.info("Slides date range: %s ~ %s", dmin, dmax)
        if stats.invalid_files:
            preview = stats.invalid_files[: min(5, len(stats.invalid_files))]
            logger.warning("Slides invalid filename examples: %s", "; ".join(preview))

    def _log_images_summary(self, stats: ImagesStats) -> None:
        logger.info("Scanned images at %s", stats.base_dir)
        logger.info("Total images: %d", stats.total_images)
        if stats.by_ext:
            items = ", ".join(f"{k}: {v}" for k, v in sorted(stats.by_ext.items()))
            logger.info("By ext: %s", items)
        if stats.by_source_from_pptname:
            items = ", ".join(
                f"{k}: {v}" for k, v in sorted(stats.by_source_from_pptname.items())
            )
            logger.info("By source(from ppt_name): %s", items)
        if stats.ppt_to_images:
            # Show top 5 items
            preview_items = list(sorted(stats.ppt_to_images.items()))[:5]
            preview = "; ".join(f"{ppt}: {cnt}" for ppt, cnt in preview_items)
            logger.info("Per-PPT image counts (sample): %s", preview)
        if stats.invalid_files:
            preview = stats.invalid_files[: min(5, len(stats.invalid_files))]
            logger.warning("Images invalid filename examples: %s", "; ".join(preview))

    # -------------------- HTML report --------------------
    def slides_to_html(self) -> str:
        """
        Generate HTML report for slides
        """
        if self.slides_stats is None:
            return "<p>No slides stats. Call inspect_slides() first.</p>"
        s = self.slides_stats
        def e(x: str) -> str:
            return escape(str(x))
        parts = []
        parts.append("<h2>Slides 概览</h2>")
        parts.append("<table>")
        parts.append(f"<tr><th>路径</th><td class='mono'>{e(s.base_dir)}</td></tr>")
        parts.append(f"<tr><th>总 PPTX 数</th><td>{s.total_pptx}</td></tr>")
        dmin, dmax = s.date_range()
        parts.append(f"<tr><th>日期范围</th><td>{e(dmin or '-') } ~ {e(dmax or '-')}</td></tr>")
        if s.by_source_count:
            dist = ", ".join(f"{e(k)}: {v}" for k, v in sorted(s.by_source_count.items()))
            parts.append(f"<tr><th>来源分布</th><td>{dist}</td></tr>")
        parts.append("</table>")
        if s.invalid_files:
            parts.append("<h3>无效文件名示例</h3><ul>")
            for path in s.invalid_files[: min(10, len(s.invalid_files))]:
                parts.append(f"<li class='mono'>{e(path)}</li>")
            parts.append("</ul>")
        return "".join(parts)

    def images_to_html(self) -> str:
        """
        Generate HTML report for images
        """
        if self.images_stats is None:
            return "<p>No images stats. Call inspect_images() first.</p>"
        s = self.images_stats
        def e(x: str) -> str:
            return escape(str(x))
        parts = []
        parts.append("<h2>Extracted Images 概览</h2>")
        parts.append("<table>")
        parts.append(f"<tr><th>路径</th><td class='mono'>{e(s.base_dir)}</td></tr>")
        parts.append(f"<tr><th>总图片数</th><td>{s.total_images}</td></tr>")
        if s.by_ext:
            dist = ", ".join(f"{e(k)}: {v}" for k, v in sorted(s.by_ext.items()))
            parts.append(f"<tr><th>扩展名分布</th><td>{dist}</td></tr>")
        if s.by_source_from_pptname:
            dist = ", ".join(f"{e(k)}: {v}" for k, v in sorted(s.by_source_from_pptname.items()))
            parts.append(f"<tr><th>来源分布(基于ppt名)</th><td>{dist}</td></tr>")
        parts.append("</table>")
        if s.ppt_to_images:
            parts.append("<h3>按 PPT 汇总（样例）</h3>")
            parts.append("<table><tr><th>PPT 名</th><th>图片数</th></tr>")
            for ppt, cnt in list(sorted(s.ppt_to_images.items()))[:10]:
                parts.append(f"<tr><td class='mono'>{e(ppt)}</td><td>{cnt}</td></tr>")
            parts.append("</table>")
        if s.invalid_files:
            parts.append("<h3>无效图片文件名示例</h3><ul>")
            for path in s.invalid_files[: min(10, len(s.invalid_files))]:
                parts.append(f"<li class='mono'>{e(path)}</li>")
            parts.append("</ul>")
        return "".join(parts)

    def to_html(self) -> str:
        """
        Generate HTML report
        """
        parts = []
        parts.append("<html><head><meta charset='utf-8'><title>PPT Report</title>")
        parts.append("<style>body{font-family:Arial,Helvetica,sans-serif;}")
        parts.append("h1,h2,h3{margin:0.6em 0;} table{border-collapse:collapse;width:100%;margin:10px 0;}")
        parts.append("th,td{border:1px solid #ddd;padding:6px;text-align:left;}")
        parts.append("tr:nth-child(even){background:#fafafa;} .muted{color:#777;} .mono{font-family:monospace;}")
        parts.append("</style></head><body>")
        parts.append("<h1>PPT 数据报告</h1>")
        parts.append(self.slides_to_html())
        parts.append(self.images_to_html())
        parts.append("</body></html>")
        return "".join(parts)

    def save_html(self, filepath: str) -> None:
        """
        Save HTML report to a file
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.to_html())

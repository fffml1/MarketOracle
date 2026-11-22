"""Page extraction and evidence assembly for the offline experiment."""
import re
from collections import Counter
from dataclasses import asdict, dataclass, field, replace

from .source import NUMERIC
from .tables import (Cell, FURNITURE, HEADING, MARKED_UNIT, STATEMENT,
                     bounds, clip, compact, extract_table, overlaps)

TABLE_SETTINGS = {"vertical_strategy": "lines_strict", "horizontal_strategy": "lines_strict"}
SENTENCE = re.compile(r"(?<=[。！？；])")
CORRECTION_REASON = re.compile(r"更正的(?:主要)?原因")
SPECIAL_HEADING = re.compile(r"^重要内容提示[:：]?$")
FINANCIAL_HEADING = re.compile(r"会计数据|财务指标|财务数据|财务报表|利润表|资产负债表|现金流量表|非经常性损益")
PERIOD_CONTEXT = re.compile(r"20\d{2}\s*年.{0,7}(?:季度|年度|半年)")


@dataclass
class PageResult:
    number: int
    status: str = "无正文"
    blocks: list = field(default_factory=list)
    skipped: list = field(default_factory=list)
    tables: list = field(default_factory=list)
    candidates: int = 0
    accepted: int = 0
    finance_hint: bool = False
    top_header: object = None
    last_header: object = None


def lines(page, chars):
    filtered = page.filter(lambda obj: obj.get("object_type") != "char" or id(obj) in chars)
    return [
        Cell(line["text"].strip(), (line["x0"], line["top"], line["x1"], line["bottom"]), page.page_number)
        for line in filtered.extract_text_lines(return_chars=False, x_tolerance=2)
        if line["text"].strip()
    ]


def page_tables(page):
    ruled = page.find_tables(TABLE_SETTINGS)
    return sorted(
        [*ruled, *(table for table in page.find_tables()
                   if not any(overlaps(table.bbox, other.bbox) for other in ruled))],
        key=lambda table: (table.bbox[1], table.bbox[0]),
    )


def position(cell):
    return asdict(cell) if cell else None


def fragment(source, kind, text, locations, section, **extra):
    return {
        "kind": kind, "text": text, "locations": locations, "section": section,
        "announcement_id": source.id, "stock_code": source.code, "company": source.company,
        "category": source.category, "period": source.period, "version": source.version,
        "disclosed_on": source.date, "sha256": source.sha256, "source_url": source.url,
        "file": source.file, "page_start": min(loc["page"] for loc in locations),
        "page_end": max(loc["page"] for loc in locations), **extra,
    }


def location(page, box):
    return {"page": page, "bbox": box}


def paragraphs(body, previous, initial_heading):
    groups = []
    heading = initial_heading
    active = []
    for line in body:
        text = compact(line.text)
        if (HEADING.match(text) or STATEMENT.fullmatch(text) or SPECIAL_HEADING.fullmatch(text)) and len(text) <= 85:
            if active:
                groups.append((heading, active))
                active = []
            heading = line.text
        last = active[-1] if active else None
        if last and (line.bbox[1] - last.bbox[3] > 11
                     or (last.text.endswith(("。", "！", "？", "；"))
                         and line.bbox[0] > min(item.bbox[0] for item in active) + 8)):
            groups.append((heading, active))
            active = []
        if active and any(last.bbox[1] < box[1] and line.bbox[1] >= box[3] for box in previous):
            groups.append((heading, active))
            active = []
        active.append(line)
    if active:
        groups.append((heading, active))
    return groups, heading


def split_long(group):
    text = "".join(line.text for line in group)
    if len(text) <= 800:
        return [(text, group)]
    chunks = []
    buffer = ""
    for sentence in SENTENCE.split(text):
        if buffer and len(buffer) + len(sentence) > 800:
            chunks.append((buffer, group))
            buffer = ""
        # Long runs without sentence punctuation still have a fixed upper bound.
        while len(sentence) > 800:
            chunks.append((sentence[:800], group))
            sentence = sentence[800:]
        buffer += sentence
    if buffer:
        chunks.append((buffer, group))
    return chunks


def body_fragments(source, groups):
    section, buffer, size = "", [], 0
    for heading, group in groups:
        for text, line_group in split_long(group):
            if buffer and (section != heading or size + len(text) > 800
                           or source.category != "定期报告" or size >= 300):
                yield body_fragment(source, section, buffer)
                buffer, size = [], 0
            buffer.append((text, line_group))
            section, size = heading, size + len(text)
    if buffer:
        yield body_fragment(source, section, buffer)


def body_fragment(source, section, parts):
    by_page = {}
    for _, lines_in_part in parts:
        for line in lines_in_part:
            by_page.setdefault(line.page, []).append(line.bbox)
    locations = [location(page, bounds(boxes)) for page, boxes in by_page.items()]
    context = [source.company, source.period, source.title, section]
    body = "\n".join(text for text, _ in parts)
    return fragment(source, "正文", " | ".join(dict.fromkeys(item for item in context if item)) + "\n" + body,
                    locations, section, original_text=body)


def row_fragment(source, box, metric, header, fields, reason_lines):
    label = header.table.text or (header.scope.text if header.scope else "")
    report_period = source.period
    if source.category == "补充更正" and header.scope:
        match = PERIOD_CONTEXT.search(header.scope.text)
        report_period = compact(match[0]) if match else report_period
    locations = [location(metric.page, box)]
    for page in sorted({column.page for column in header.labels} - {metric.page}):
        locations.append(location(page, bounds([col.bbox for col in header.labels if col.page == page])))
    if reason_lines:
        locations.extend(location(line.page, line.bbox) for line in reason_lines)
    content = " | ".join(dict.fromkeys(item for item in
                                       (source.company, report_period, source.title,
                                        header.scope.text if header.scope else "", label, metric.text) if item))
    parts = [f"{field['header']['text']}：{field['value']['text']}" +
             (f" [{field['unit']}]" if "unit" in field else "") for field in fields]
    if reason_lines:
        parts.append("公告说明的更正原因：" + "".join(line.text for line in reason_lines))
    return fragment(
        source, "更正行" if source.category == "补充更正" else "表格行",
        content + "\n" + "\n".join(parts), locations, label,
        period=report_period, metric=position(metric), header_page=header.labels[0].page,
        value_page=metric.page, table=position(header.table), scope=position(header.scope),
        fields=fields, correction_reason=[position(line) for line in reason_lines],
    )


class Extractor:
    def __init__(self, source):
        self.source = source
        self.previous_number = 0
        self.header = None
        self.scope = None
        self.section = ""
        self.reason = []
        self.previous_body_ended_at_margin = False
        self.body = []

    def page(self, page):
        if page.page_number != self.previous_number + 1:
            self.header, self.scope = None, None
            self.section = ""
            self.previous_body_ended_at_margin = False
            self.reason.clear()
        result = PageResult(page.page_number)
        self.previous_number = page.page_number
        tables = page_tables(page)
        result.tables = [table.bbox for table in tables]
        outside = {
            id(char) for char in page.chars if not any(
                box[0] <= (char["x0"] + char["x1"]) / 2 < box[2]
                and box[1] <= (char["top"] + char["bottom"]) / 2 < box[3] for box in result.tables
            )
        }
        body = lines(page, outside)
        body = [line for line in body if not (
            (line.bbox[1] < page.height * 0.11 or line.bbox[1] > page.height * 0.87)
            and FURNITURE.fullmatch(line.text)
        )]
        printable = sum("\u3400" <= ch <= "\u9fff" or ch.isascii() and ch.isalnum()
                        for char in page.chars for ch in char["text"])
        if printable < 10 and (page.images or len(page.chars) > 15):
            result.skipped.append((page.bbox, "P01_文字层不足_待OCR"))
            result.status = "全部跳过"
            self.header = None
            self.previous_body_ended_at_margin = False
            return result

        gap_lines = []
        start = 0
        correction = self.source.category == "补充更正"
        unit = None
        for index, table in enumerate(tables):
            gap_lines = [line for line in body if start <= line.bbox[1] and line.bbox[3] <= table.bbox[1] + 2]
            if gap_lines:
                unit = next((line for line in reversed(gap_lines)
                             if MARKED_UNIT.search(compact(line.text))
                             and table.bbox[1] - line.bbox[3] <= 35), None)
            else:
                unit = None
            caption = None
            for line in gap_lines:
                title = compact(line.text)
                if HEADING.match(title) or STATEMENT.fullmatch(title):
                    caption = line
                    if not correction or PERIOD_CONTEXT.search(title):
                        self.scope = line
            previous = self.header if index == 0 else None
            try:
                records, skipped, header = extract_table(
                    table, previous, "\n".join(line.text for line in gap_lines), unit, self.scope, correction)
            except (ValueError, TypeError, IndexError, KeyError) as error:
                result.skipped.append((table.bbox, "T09_单块解析失败", f"{type(error).__name__}: {error}"))
                self.header = None
                start = table.bbox[3]
                continue
            if caption:
                records = [(box, metric, replace(row_header, table=caption), fields)
                           for box, metric, row_header, fields in records]
                if header:
                    header = replace(header, table=caption)
            if index == 0:
                result.top_header = header
            self.header = header
            result.accepted += len(records)
            result.candidates += len(records) + len(skipped)
            result.blocks.extend(row_fragment(self.source, *row, self.reason if correction else ()) for row in records)
            result.skipped.extend(skipped)
            if not records and not skipped:
                result.skipped.append((table.bbox, "T08_表格无候选数值"))
            start = table.bbox[3]

        result.last_header = self.header if tables else None
        # A section or paragraph after the last table ends its continuation context.
        if not tables or any(line.bbox[1] > start + 3 and not MARKED_UNIT.search(compact(line.text))
                             for line in body):
            self.header = None

        result.finance_hint = bool(FINANCIAL_HEADING.search(compact(page.extract_text() or "")))
        clean_body = []
        for line in body:
            if (self.source.category in {"定期报告", "补充更正"} and
                len(NUMERIC.findall(line.text)) >= 2 and not re.search(r"[。；：]", line.text)):
                result.skipped.append((line.bbox, "P02_疑似表外数字行"))
            elif not MARKED_UNIT.search(compact(line.text)) or not tables:
                clean_body.append(line)
        groups, self.section = paragraphs(clean_body, result.tables, self.section)
        if correction:
            for heading, group in groups:
                if CORRECTION_REASON.search(heading):
                    self.reason.extend(line for line in group if line.text != heading)
        if (groups and self.body and self.previous_body_ended_at_margin
            and groups[0][0] == self.body[-1][0] and groups[0][1][0].bbox[1] < page.height * 0.20
            and not any(box[1] < groups[0][1][0].bbox[1] for box in result.tables)
            and not HEADING.match(compact(groups[0][1][0].text))):
            self.body[-1][1].extend(groups[0][1])
            retained_groups = groups[1:]
        else:
            retained_groups = groups
        self.body.extend(retained_groups)
        self.previous_body_ended_at_margin = bool(
            clean_body and clean_body[-1].bbox[3] > page.height * 0.76
            and not re.search(r"[。！？：；]$", clean_body[-1].text)
            and not any(box[3] > clean_body[-1].bbox[3] for box in result.tables)
        )
        if result.skipped:
            result.status = "部分解析" if result.blocks or groups else "全部跳过"
        elif result.blocks or groups:
            result.status = "已全部解析"
        return result


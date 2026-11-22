"""Conservative, position-based validation of financial PDF tables."""
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass

METRICS = {
    "营业收入": r"^(?:其中[:：])?营业(?:总)?收入(?:[（(].*)?$",
    "归母净利润": r"^归属于(?:上市公司|母公司)(?:所有者|股东)的?净利润(?:[（(].*)?$",
    "经营现金流量净额": r"^经营活动产生的现金流量净额(?:[（(].*)?$",
}
UNIT_TOKEN = r"(?:万元/股|元/股|人民币(?:亿元|万元|千元|元)|亿元|百万元|万元|千元|元|个百分点|%|万股|股|倍)"
MARKED_UNIT = re.compile(r"单位\s*[:：]\s*(" + UNIT_TOKEN + ")")
LABEL_UNIT = re.compile(r"[（(](" + UNIT_TOKEN + r")[）)]")
NUMBER = re.compile(r"[-+−]?\s*(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?")
FOOTNOTE = re.compile(r"\(注[0-9一二三四五六七八九十]*\)$")
MISSING = {"", "-", "--", "—", "——", "–", "/", "不适用", "N/A"}
CHANGE = re.compile(r"增减|变[动化](?:率|幅度)|同比|环比|增长率")
PERIOD_LABEL = re.compile(r"本报告期|上年同期|上年度末|上年末|报告期(?:末|初)|本期|上期|"
                          r"本年|上年|年初|年末|期末|期初|年度|20\d{2}(?:年|[-./])")
CORRECTION = re.compile(r"原列报|原披露|更正[前后]|调整金额|重述[前后]|调整[前后]")
NOTE = re.compile(r"说明|原因|附注|备注")
HEADING = re.compile(r"^(?:第[一二三四五六七八九十]+[章节节]|[一二三四五六七八九十]+[、.]|"
                     r"[（(][一二三四五六七八九十]+[）)]|\d{1,2}[.、](?!\d))\s*\S")
STATEMENT = re.compile(r"^(?:合并|母公司)(?:资产负债表|利润表|现金流量表)(?:[（(].*)?$")
FURNITURE = re.compile(r"^\s*(?:\d{1,3}(?:\s*/\s*\d{1,3})?|.{2,55}20\d\d\s*年.{0,12}报告(?:全文)?)\s*$")
BBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class Cell:
    text: str
    bbox: BBox
    page: int


@dataclass(frozen=True)
class Header:
    labels: tuple[Cell, ...]
    layout: tuple[tuple[float, float], ...]
    unit: Cell | None
    scope: Cell | None
    table: Cell


@dataclass(frozen=True)
class Band:
    bbox: BBox
    values: tuple[Cell, ...]


def compact(text):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))


def numeric(text):
    text = FOOTNOTE.sub("", compact(text))
    core = text[1:-1] if text.startswith("(") and text.endswith(")") else text
    return bool(NUMBER.fullmatch(core) or re.fullmatch(r"(?:增加|减少)\d+(?:\.\d+)?个百分点", core))


def number_like(text):
    return (numeric(text) or bool(re.match(r"^[-+−(]?[\d０-９][\d.,，．%％()\-+−]*$", compact(text))))


def midpoint(bbox):
    return (bbox[0] + bbox[2]) / 2


def horizontal_match(bbox, span):
    return abs(bbox[0] - span[0]) <= 3 and abs(bbox[2] - span[1]) <= 3


def overlaps(left, right):
    return (min(left[2], right[2]) > max(left[0], right[0])
            and min(left[3], right[3]) > max(left[1], right[1]))


def bounds(boxes):
    from pdfplumber import utils

    return utils.merge_bboxes(boxes)


def clip(chars, bbox, page):
    from pdfplumber import utils

    selected = [
        char for char in chars
        if bbox[0] <= (char["x0"] + char["x1"]) / 2 < bbox[2]
        and bbox[1] <= (char["top"] + char["bottom"]) / 2 < bbox[3]
    ]
    text = utils.extract_text(selected, x_tolerance=2) if selected else ""
    return Cell(text.strip(), utils.objects_to_bbox(selected) if selected else bbox, page)


def locate_header(labels, layout, unit, scope, table, *, correction=False):
    columns = [compact(label.text) for label in labels]
    if not columns or len(set(columns)) != len(columns):
        return None
    time_label = CORRECTION if correction else PERIOD_LABEL
    if not any(time_label.search(label) for label in columns):
        return None
    if correction and (scope is None or not re.search(r"20\d{2}年", compact(scope.text))):
        return None
    if any(not (time_label.search(label) or CHANGE.search(label) or NOTE.search(label))
           for label in columns):
        return None
    return Header(tuple(labels), tuple(layout), unit, scope, table)


def validate_row(metric, values, header):
    if not compact(metric.text) or not re.search(r"[\u3400-\u9fffA-Za-z]", metric.text):
        return None, "T01_指标缺失"
    if header is None:
        return None, "T02_表头或口径不明"
    if len(values) != len(header.labels) or any(
        not horizontal_match(value.bbox, span) for value, span in zip(values, header.layout)
    ):
        return None, "T03_列位置不匹配"
    metric_unit = LABEL_UNIT.search(compact(metric.text))
    fields = []
    for value, label in zip(values, header.labels):
        column = compact(label.text)
        literal = FOOTNOTE.sub("", compact(value.text))
        if NOTE.search(column) and not (PERIOD_LABEL.search(column) or CORRECTION.search(column)):
            fields.append({"value": asdict(value), "header": asdict(label), "role": "说明"})
            continue
        column_unit = LABEL_UNIT.search(column) or MARKED_UNIT.search(column)
        unit_source = None
        change = CHANGE.search(column)
        declared_units = {match[1] for match in LABEL_UNIT.finditer(column)}
        declared_units.update(match[1] for match in MARKED_UNIT.finditer(column))
        if len(declared_units) > 1 or (
            not change and column_unit and metric_unit and column_unit[1] != metric_unit[1]
        ):
            return None, "T05_数值单位冲突"
        if literal.endswith("个百分点"):
            unit, unit_source = "个百分点", value
        elif literal.endswith("%") or (change and not column_unit):
            if change and literal not in MISSING and not literal.endswith("%") and not column_unit:
                return None, "T04_变化列缺少单位"
            unit, unit_source = "%", value if literal.endswith("%") else label
        elif column_unit:
            unit, unit_source = column_unit[1], label
        elif metric_unit:
            unit, unit_source = metric_unit[1], metric
        elif header.unit:
            unit, unit_source = MARKED_UNIT.search(compact(header.unit.text))[1], header.unit
        else:
            return None, "T04_数值缺少单位"
        if change and unit not in {"%", "个百分点"}:
            return None, "T05_变化列单位冲突"
        if literal.endswith("%") and not change and (
            (column_unit and column_unit[1] != "%")
            or (metric_unit and metric_unit[1] != "%")
            or (not column_unit and not metric_unit and header.unit
                and MARKED_UNIT.search(compact(header.unit.text))[1] != "%")
        ):
            return None, "T05_数值单位冲突"
        if literal not in MISSING and not numeric(value.text):
            return None, "T06_数值格式或错列"
        fields.append({
            "value": asdict(value), "header": asdict(label), "unit": unit,
            "unit_source": asdict(unit_source),
            "role": "缺项" if literal in MISSING else "数值",
        })
    if not any(field["role"] == "数值" for field in fields):
        return None, "T07_无可用数值"
    return fields, None


def table_bands(table):
    data = {}
    for row, texts in zip(table.rows, table.extract(x_tolerance=2)):
        for box, value in zip(row.cells, texts):
            if box and value and number_like(value):
                data.setdefault((box[1], box[3]), []).append(Cell(value, box, table.page.page_number))
    # Some Word-exported PDFs create a second grid inside the metric cell. The amount
    # cells can still define the row, including all the wrapped lines in its metric.
    return [
        Band((table.bbox[0], top, table.bbox[2], bottom), tuple(sorted(values, key=lambda c: c.bbox[0])))
        for (top, bottom), values in sorted(data.items())
    ]


def layout_from(bands, box):
    layouts = Counter(
        tuple(round(value.bbox[0], 1) for value in band.values) for band in bands
        if band.values[0].bbox[0] > box[0] + 15
    )
    if not layouts:
        return ()
    starts = max(layouts, key=lambda layout: (len(layout), layouts[layout]))
    return tuple(zip(starts, (*starts[1:], box[2])))


def layout_for_header(table, bands, top, first):
    layout = layout_from(bands, table.bbox)
    if not layout:
        return ()
    # A text/notes column may follow the last amount. Take its boundary from the
    # actual cells in the numeric row so it cannot be swallowed by the amount.
    last = layout[-1]
    candidates = sorted({
        cell[2] for row in table.rows for cell in row.cells if cell
        and abs(cell[0] - last[0]) <= 3 and cell[2] < last[1] - 12
        and abs(cell[1] - first.bbox[1]) <= 1 and abs(cell[3] - first.bbox[3]) <= 1
    })
    if not candidates:
        return layout
    end = candidates[-1]
    chars = table.page.chars
    note = clip(chars, (end, top, last[1], first.bbox[1]), table.page.page_number)
    if NOTE.search(compact(note.text)):
        return (*layout[:-1], (last[0], end), (end, last[1]))
    return layout


def read_header(table, bands, top, first, unit, scope, correction):
    layout = layout_for_header(table, bands, top, first)
    page = table.page
    if not layout:
        return None
    # Multiple independent columns before the first numeric column are ambiguous
    # for the supported single-metric row format.
    leading_labels = [
        cell for row in table.rows for cell in row.cells
        if cell and cell[2] <= layout[0][0] + 1 and top <= cell[1] < cell[3] <= first.bbox[1] + 0.1
        and len(compact(clip(page.chars, cell, page.page_number).text)) >= 2
    ]
    if any(left[2] <= right[0] and max(left[1], right[1]) < min(left[3], right[3])
           for left in leading_labels for right in leading_labels):
        return None
    box = (table.bbox[0], top, table.bbox[2], first.bbox[1])
    table_label = clip(page.chars, (box[0], box[1], layout[0][0], box[3]),
                       page.page_number)
    if not table_label.text and scope:
        table_label = scope
    labels = tuple(
        clip(page.chars, (left, top, right, first.bbox[1]), page.page_number)
        for left, right in layout
    )
    # A real spanning parent period can qualify several subordinate headers.
    # Restrict it to geometrically enclosed columns and preserve the parent source.
    merged = [
        cell for row in table.rows for cell in row.cells
        if cell and top <= cell[1] < cell[3] <= first.bbox[1] + 0.1
        and sum(cell[0] - 2 <= left and right <= cell[2] + 2 for left, right in layout) > 1
        and cell[3] < first.bbox[1] - 2
    ]
    for cell in dict.fromkeys(merged):
        parent = clip(page.chars, cell, page.page_number)
        if PERIOD_LABEL.search(compact(parent.text)) and not CHANGE.search(compact(parent.text)):
            labels = tuple(
                Cell(parent.text + "\n" + child.text, bounds([parent.bbox, child.bbox]), child.page)
                if cell[0] - 2 <= left and right <= cell[2] + 2
                and (child := clip(page.chars, (left, cell[3], right, first.bbox[1]), page.page_number)).text
                else label
                for label, (left, right) in zip(labels, layout)
            )
    return locate_header(labels, layout, unit, scope, table_label, correction=correction)


def continuation(previous, table, bands, gap, unit, scope):
    if not previous or len(bands) < 2 or (gap.strip() and not FURNITURE.fullmatch(gap.strip())):
        return None
    layout = layout_from(bands, table.bbox)
    if (table.bbox[1] > 0.22 * table.page.height or not layout
        or unit and (not previous.unit or compact(unit.text) != compact(previous.unit.text))
        or scope and previous.scope and compact(scope.text) != compact(previous.scope.text)):
        return None
    if not any(label.page == table.page.page_number - 1 for label in previous.labels):
        return None
    old_starts = tuple(left for (left, _), label in zip(previous.layout, previous.labels)
                       if not NOTE.search(compact(label.text)))
    if len(old_starts) != len(layout) or any(abs(old - new[0]) > 3 for old, new in zip(old_starts, layout)):
        return None
    if abs(previous.layout[-1][1] - table.bbox[2]) > 3:
        return None
    return previous


def extract_table(table, previous, gap, unit, scope, correction):
    page = table.page
    bands = table_bands(table)
    records, skipped = [], []
    if not bands:
        return records, skipped, None
    header = None
    top = table.bbox[1]
    for index, band in enumerate(bands):
        between = clip(page.chars, (table.bbox[0], top, table.bbox[2], band.bbox[1]), page.page_number)
        if between.text:
            header = read_header(table, bands[index:], top, band, unit, scope, correction)
        elif index == 0:
            header = continuation(previous, table, bands, gap, unit, scope)
        top = max(top, band.bbox[3])
        if not header:
            skipped.append((band.bbox, "T02_表头或口径不明"))
            continue
        if any(sum(horizontal_match(value.bbox, span) for span in header.layout) != 1
               for value in band.values):
            skipped.append((band.bbox, "T03_列位置不匹配"))
            continue
        metric = clip(page.chars, (band.bbox[0], band.bbox[1], header.layout[0][0], band.bbox[3]),
                      page.page_number)
        values = []
        source_cells = []
        for left, right in header.layout:
            cell_box = (left, band.bbox[1], right, band.bbox[3])
            cell = clip(page.chars, cell_box, page.page_number)
            values.append(Cell(cell.text, cell_box, page.page_number))
            source_cells.append(cell)
        fields, reason = validate_row(metric, values, header)
        if reason:
            skipped.append((band.bbox, reason))
        else:
            for field, source_cell in zip(fields, source_cells):
                field["cell_bbox"] = field["value"]["bbox"]
                field["value"] = asdict(source_cell)
            records.append((band.bbox, metric, header, fields))
    return records, skipped, header

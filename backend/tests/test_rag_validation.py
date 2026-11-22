"""Stage 1 acceptance checks for financial rows and source citations."""
import unittest
from dataclasses import replace
from types import SimpleNamespace

try:
    import pdfplumber
except ModuleNotFoundError as error:
    raise unittest.SkipTest("离线 PDF 验收需要 pdfplumber") from error

from backend.rag.extract import Extractor, body_fragments, paragraphs
from backend.rag.source import INDEX, read_index, select_version
from backend.rag.tables import (Band, Cell, compact, continuation, extract_table,
                                locate_header, validate_row)


def cell(text, left=0, right=80, *, page=1, top=20, bottom=32):
    return Cell(text, (left, top, right, bottom), page)


def positioned_table(rows):
    chars = []
    for row in rows:
        for item in row:
            for index, character in enumerate(item.text):
                x = item.bbox[0] + 2 + index * 5
                y = item.bbox[1] + 2
                chars.append({"text": character, "x0": x, "x1": x + 5,
                              "top": y, "bottom": y + 8, "doctop": y, "upright": True})
    return SimpleNamespace(
        bbox=(0, rows[0][0].bbox[1], rows[0][-1].bbox[2], rows[-1][0].bbox[3]),
        page=SimpleNamespace(chars=chars, page_number=1, height=840),
        rows=[SimpleNamespace(cells=[item.bbox for item in row]) for row in rows],
        extract=lambda **_: [[item.text for item in row] for row in rows],
    )


class FinancialRulesTests(unittest.TestCase):
    def setUp(self):
        self.layout = ((80, 140), (140, 200), (200, 260), (260, 320))
        self.labels = tuple(cell(label, *span) for label, span in zip(
            ("本报告期", "本报告期比上年同期增减", "年初至报告期末", "年初至报告期末同比(%)"),
            self.layout,
        ))
        self.unit = cell("单位：千元", page=1)
        self.metric = cell("营业收入", page=1, top=34, bottom=46)
        self.values = tuple(cell(text, *span, top=34, bottom=46) for text, span in zip(
            ("-1,250.00", "−2.15%", "(2,000.00)", "0"), self.layout,
        ))
        self.header = self.make_header()

    def make_header(self, labels=None, unit=True):
        return locate_header(labels or self.labels, self.layout, self.unit if unit else None,
                             cell("主要财务数据"), cell("主要会计数据"))

    def test_quarter_and_cumulative_stay_together_with_units(self):
        fields, reason = validate_row(self.metric, self.values, self.header)
        self.assertIsNone(reason)
        self.assertEqual([field["value"]["text"] for field in fields],
                         ["-1,250.00", "−2.15%", "(2,000.00)", "0"])
        self.assertEqual([field["unit"] for field in fields], ["千元", "%", "千元", "%"])
        self.assertEqual([field["header"]["text"] for field in fields], [item.text for item in self.labels])
        self.assertTrue(all(field["unit_source"]["page"] == field["value"]["page"] == 1 for field in fields))

    def test_generated_table_passes_same_coordinate_parser_as_pdf(self):
        table = positioned_table([(cell("项目"), *self.labels), (self.metric, *self.values)])
        records, skipped, _ = extract_table(table, None, "", self.unit, self.header.scope, False)
        self.assertEqual(skipped, [])
        self.assertEqual(len(records), 1)
        _, metric, _, fields = records[0]
        self.assertEqual(metric.text, "营业收入")
        self.assertEqual([field["unit"] for field in fields], ["千元", "%", "千元", "%"])
        self.assertEqual([field["value"]["text"] for field in fields],
                         ["-1,250.00", "−2.15%", "(2,000.00)", "0"])
        self.assertTrue(all(field["value"]["bbox"] != field["cell_bbox"] for field in fields))
        records, skipped, _ = extract_table(table, None, "", None, self.header.scope, False)
        self.assertEqual(records, [])
        self.assertEqual(skipped[0][1], "T04_数值缺少单位")

    def test_percentage_footnote_preserves_the_original_cell(self):
        marked = (self.values[0], replace(self.values[1], text="82.16%（注1）"), *self.values[2:])
        fields, reason = validate_row(self.metric, marked, self.header)
        self.assertIsNone(reason)
        self.assertEqual(fields[1]["role"], "数值")
        self.assertEqual(fields[1]["unit"], "%")
        self.assertEqual(fields[1]["value"]["text"], "82.16%（注1）")

    def test_metric_unit_and_missing_literals_are_preserved(self):
        values = (replace(self.values[0], text="—"), replace(self.values[1], text=""), *self.values[2:])
        fields, reason = validate_row(cell("营业收入（元）"), values, self.header)
        self.assertIsNone(reason)
        self.assertEqual([(item["value"]["text"], item["unit"], item["role"]) for item in fields],
                         [("—", "元", "缺项"), ("", "%", "缺项"), ("(2,000.00)", "元", "数值"), ("0", "%", "数值")])

    def test_removed_unit_ambiguous_header_and_swapped_columns_skip(self):
        self.assertEqual(validate_row(self.metric, self.values, self.make_header(unit=False))[1],
                         "T04_数值缺少单位")
        duplicate = (*self.labels[:2], replace(self.labels[2], text=self.labels[0].text), self.labels[3])
        self.assertIsNone(self.make_header(duplicate))
        self.assertEqual(validate_row(self.metric, self.values, None)[1], "T02_表头或口径不明")
        self.assertEqual(validate_row(self.metric, (self.values[1], self.values[0], *self.values[2:]),
                                      self.header)[1], "T03_列位置不匹配")
        swapped = (replace(self.values[0], text=self.values[1].text),
                   replace(self.values[1], text=self.values[0].text), *self.values[2:])
        self.assertEqual(validate_row(self.metric, swapped, self.header)[1], "T05_数值单位冲突")

    def test_continuation_requires_matching_geometry_context_and_units(self):
        second_values = tuple(replace(value, page=2) for value in self.values)
        bands = [Band((0, 30, 320, 46), second_values), Band((0, 46, 320, 62), second_values)]
        table = SimpleNamespace(bbox=(0, 30, 320, 250), page=SimpleNamespace(height=840, page_number=2))
        header = continuation(self.header, table, bands, "", None, self.header.scope)
        fields, reason = validate_row(replace(self.metric, page=2), second_values, header)
        self.assertIsNone(reason)
        self.assertEqual({(field["header"]["page"], field["value"]["page"]) for field in fields}, {(1, 2)})
        self.assertIsNone(continuation(self.header, table, bands, "二、另一个小节", None, None))
        self.assertIsNone(continuation(self.header, table, bands, "", cell("单位：元", page=2), None))
        shifted = tuple(replace(value, bbox=(value.bbox[0] + 8, *value.bbox[1:])) for value in second_values)
        self.assertIsNone(continuation(self.header, table, [replace(band, values=shifted) for band in bands],
                                       "", None, None))

    def test_merge_heading_scope_keeps_sections_separate(self):
        lines = [
            cell("一、主要财务数据"), cell("第一段。", top=40, bottom=52),
            cell("二、股东信息", top=60, bottom=72), cell("第二段。", top=80, bottom=92),
        ]
        groups, section = paragraphs(lines, [], "")
        self.assertEqual(section, "二、股东信息")
        self.assertEqual([heading for heading, _ in groups], ["一、主要财务数据", "二、股东信息"])


@unittest.skipUnless(INDEX.is_file(), "需要本地公告样本")
class ActualSampleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sources = read_index(INDEX)

    def extract(self, aid, pages=(1, 2)):
        source = self.sources[aid]
        if not (INDEX.parent / source.file).is_file():
            self.skipTest(f"缺少本地样本 {aid}")
        extractor = Extractor(source)
        with pdfplumber.open(INDEX.parent / source.file) as pdf:
            extracted = [extractor.page(pdf.pages[page - 1]) for page in pages]
        return extracted, list(body_fragments(source, extractor.body))

    def test_version_cutoff_and_exact_values_with_no_duplicate_body(self):
        for as_of, expected, values in (
            ("2025-11-01", "1224774300", ["8,174,336,700.05", "60,945,321,083.57"]),
            ("2026-05-01", "1225273123", ["7,127,755,606.32", "30,637,727,654.97"]),
        ):
            with self.subTest(as_of=as_of):
                source = select_version(self.sources.values(), "000858", "2025年三季度", "定期报告", as_of)
                self.assertEqual(source.id, expected)
                pages, body = self.extract(source.id, (1,))
                revenue = [row for row in pages[0].blocks if compact(row["metric"]["text"]) == "营业收入(元)"]
                self.assertEqual(len(revenue), 1)
                self.assertEqual([revenue[0]["fields"][index]["value"]["text"] for index in (0, 2)], values)
                self.assertEqual((revenue[0]["announcement_id"], revenue[0]["stock_code"]), (expected, "000858"))
                self.assertTrue(all(value not in block["original_text"] for value in values for block in body))

    def test_cash_dividend_keeps_tax_and_per_ten_shares_in_same_clause(self):
        _, body = self.extract("1225417855")
        self.assertTrue(any("每10股派25.796852元" in compact(block["original_text"])
                            and "含税" in block["original_text"] for block in body))

    def test_correction_keeps_period_old_new_and_reason(self):
        pages, _ = self.extract("1225273122")
        revenue = next(block for block in pages[1].blocks if "营业收入" in block["metric"]["text"])
        self.assertEqual(revenue["period"], "2025年一季度")
        self.assertEqual([field["header"]["text"] for field in revenue["fields"]],
                         ["原列报金额", "调整金额", "重述后金额"])
        self.assertIn("收入确认", compact(revenue["text"]))
        self.assertEqual((revenue["page_start"], revenue["value_page"], revenue["page_end"]), (1, 2, 2))

    def test_annual_merged_headers_and_cross_page_value(self):
        self.assertEqual(self.sources["1225114741"].period, "2025年度")
        pages, _ = self.extract("1225114741", (91, 92))
        continued = next(block for block in pages[1].blocks if block["header_page"] == 91)
        self.assertEqual(continued["value_page"], 92)
        self.assertEqual({entry["page"] for entry in continued["locations"]}, {91, 92})
        labels = [compact(field["header"]["text"]) for field in continued["fields"]]
        self.assertIn("期末余额账面余额", labels)
        self.assertIn("期初余额账面余额", labels)


if __name__ == "__main__":
    unittest.main()

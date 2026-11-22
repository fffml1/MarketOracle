"""Announcement identity and lightweight inventory of all source pages."""
import csv
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

INDEX = Path(__file__).resolve().parents[1] / "data" / "公告索引.csv"
STOCK = "000858"
OUTPUT = INDEX.parent / "rag" / "wuliangye"
PERIOD = re.compile(r"(20\d{2})\s*年(年?度|半年度|第?[一二三四]季度)")
NUMERIC = re.compile(r"(?<![\w.])[-−]?(?:\d{1,3}(?:,\d{3})+|\d+)\.\d+%?|\d{1,3}(?:,\d{3})+")
FINANCIAL = re.compile(r"营业(?:总)?收入|净利润|资产负债表|利润表|现金流量表|"
                       r"本期(?:金额|发生额)|上期(?:金额|发生额)|会计数据和财务指标")


@dataclass(frozen=True)
class Announcement:
    id: str
    code: str
    company: str
    category: str
    date: str
    title: str
    url: str
    file: str
    sha256: str

    @property
    def period(self):
        match = PERIOD.search(self.title)
        if not match:
            return ""
        year, term = match.groups()
        return f"{year}年{term.removeprefix('第').removeprefix('年')}"

    @property
    def version(self):
        return next((tag for tag in ("更新前", "更新后", "修订版") if tag in self.title), "披露版")


def read_index(index):
    with index.open(encoding="utf-8-sig", newline="") as stream:
        rows = [Announcement(**row) for row in csv.DictReader(stream)]
    result = {row.id: row for row in rows}
    if len(result) != len(rows):
        raise ValueError("公告索引含重复 ID")
    return result


def select_version(announcements, code, period, category, as_of):
    return max(
        (item for item in announcements if item.code == code and item.period == period
         and item.category == category and item.date <= as_of),
        key=lambda item: (item.date, item.version in {"更新后", "修订版"}, item.id),
        default=None,
    )


def inventory(source, root):
    import pypdfium2 as pdfium
    from pypdfium2 import raw

    path = (root / source.file).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("PDF 路径超出索引目录")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != source.sha256:
        raise ValueError("PDF SHA-256 与索引不一致")
    pages = []
    with pdfium.PdfDocument(path) as pdf:
        for index in range(len(pdf)):
            page = pdf[index]
            row = {"page": index + 1}
            try:
                text_page = page.get_textpage()
                try:
                    text = text_page.get_text_bounded()
                finally:
                    text_page.close()
                lines = text.splitlines()
                row.update(
                    characters=len(re.sub(r"\s+", "", text)),
                    numeric_rows=sum(len(NUMERIC.findall(line)) >= 2 for line in lines),
                    numbers=len(NUMERIC.findall(text)),
                    finance_hint=bool(FINANCIAL.search(re.sub(r"\s+", "", text))),
                    image=next(page.get_objects(filter=[raw.FPDF_PAGEOBJ_IMAGE]), None) is not None,
                )
            except Exception as error:
                row["error"] = f"{type(error).__name__}: {error}"
            finally:
                page.close()
            pages.append(row)
    return pages

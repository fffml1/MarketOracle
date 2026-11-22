"""Run plan 03's reproducible first-stage validation (no network or model calls)."""
import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import NamedTemporaryFile

import pdfplumber

from .extract import Extractor, body_fragments
from .source import INDEX, inventory, read_index, select_version
from .tables import METRICS, compact

PARSER = f"pdfplumber/{pdfplumber.__version__}"
RULE_VERSION = "2"
QUARTERLY = ("1224774300", "1225273123", "1224764517", "1224721971")
ANNUAL = ("1225114741", "1225273091", "1225002214")
SPECIAL = ("1225273122", "1225417855")


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_records(path, records, *, jsonl=True):
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            if jsonl:
                for record in records:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            else:
                json.dump(records, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            stream.close()
            temporary.replace(path)
        finally:
            stream.close()
            temporary.unlink(missing_ok=True)


def collect_inventory(sources, root):
    rows = {}
    for source in sources.values():
        try:
            pages = inventory(source, root)
            rows[source.id] = {"id": source.id, "sha256": source.sha256,
                               "stock_code": source.code, "file": source.file, "pages": pages}
        except Exception as error:
            rows[source.id] = {"id": source.id, "sha256": source.sha256, "stock_code": source.code,
                               "file": source.file, "pages": [],
                               "error": f"{type(error).__name__}: {error}"}
    return rows


def annual_pair(source, root, pages):
    adjacent = [(left, right) for left, right in zip(pages, pages[1:])
                if not left.get("error") and not right.get("error")]
    if not adjacent:
        raise ValueError(f"{source.id} 没有可用相邻页")
    with pdfplumber.open(root / source.file) as pdf:
        for left, right in adjacent:
            # Prefilter from the text layer. The choice is verified against actual cells.
            if left["numeric_rows"] < 2 or right["numeric_rows"] < 2:
                continue
            extractor = Extractor(source)
            first = pdf.pages[left["page"] - 1]
            second = pdf.pages[right["page"] - 1]
            try:
                a = extractor.page(first)
                if a.last_header is None or a.accepted < 1:
                    continue
                b = extractor.page(second)
                if sum(block["header_page"] == a.number for block in b.blocks) >= 2:
                    return (a.number, b.number), "年报：下一页至少两行通过续表列位置、表头与单位检查"
                if b.top_header and not b.top_header.scope and a.last_header.layout == b.top_header.layout:
                    if ([compact(col.text) for col in a.last_header.labels]
                        == [compact(col.text) for col in b.top_header.labels] and b.accepted >= 2):
                        return (a.number, b.number), "年报：下一页重复相同表头和列布局"
            finally:
                first.close()
                second.close()
    left, right = max(adjacent, key=lambda pair: (
        pair[0]["numeric_rows"] + pair[1]["numeric_rows"], -pair[0]["page"]))
    return (left["page"], right["page"]), "年报回退：无兼容续表命中，取数字行最多的相邻两页"


def create_manifest(sources, inv, root):
    selected = {}

    def add(aid, numbers, reason):
        source = sources[aid]
        for number in numbers:
            selected[aid, number] = {"announcement_id": aid, "sha256": source.sha256,
                                     "page": number, "reason": reason}

    for aid in (*QUARTERLY, *SPECIAL):
        add(aid, (1, 2), "三季报前两页" if aid in QUARTERLY else "更正或分红实施公告前两页")
    for aid in ANNUAL:
        numbers, reason = annual_pair(sources[aid], root, inv[aid]["pages"])
        add(aid, numbers, reason)
    remaining = [(aid, page) for aid, doc in sorted(inv.items()) for page in doc["pages"]
                 if (aid, page["page"]) not in selected and not page.get("error")]
    if len(remaining) < 2:
        raise ValueError("未使用的难例候选页不足")
    graphic = [(aid, page) for aid, page in remaining if page["image"] and page["characters"] <= 100]
    density = lambda item: (-item[1]["numbers"], item[0], item[1]["page"])
    if graphic:
        aid, page = min(graphic, key=lambda item: (item[1]["characters"], item[0], item[1]["page"]))
        reason = "难例：含图且文字层少于或等于100字符"
    else:
        aid, page = min(remaining, key=density)
        reason = "难例回退：无少字图页，改取数字密集页"
    add(aid, (page["page"],), reason)
    aid, page = min((item for item in remaining if (item[0], item[1]["page"]) not in selected), key=density)
    add(aid, (page["page"],), "难例：未用页面中数字最密集")
    return list(selected.values())


def validate_manifest(manifest, sources, inv):
    keys = {(row["announcement_id"], row["page"]) for row in manifest}
    if len(keys) != len(manifest) or len(keys) != 20:
        raise ValueError("样本清单必须含 20 个不同的公告及物理页码组合")
    if any((aid, page) not in keys for aid in (*QUARTERLY, *SPECIAL) for page in (1, 2)):
        raise ValueError("样本清单缺少固定公告页")
    for aid in ANNUAL:
        pages = sorted(page for doc, page in keys if doc == aid)
        if not any(right == left + 1 for left, right in zip(pages, pages[1:])):
            raise ValueError(f"{aid} 样本中缺少相邻页")
    for row in manifest:
        aid, number = row["announcement_id"], row["page"]
        if aid not in sources or row["sha256"] != sources[aid].sha256:
            raise ValueError(f"固定样本原文哈希已变化: {aid}")
        if not row["reason"] or number < 1 or (not inv[aid].get("error") and number > len(inv[aid]["pages"])):
            raise ValueError(f"固定样本页码或选取理由无效: {aid} / {number}")


def skip(source, number, bbox, rule, detail=None):
    return {"announcement_id": source.id, "stock_code": source.code, "sha256": source.sha256,
            "page": number, "bbox": bbox, "rule": rule, "detail": detail, "source_url": source.url,
            "parser": PARSER, "rule_version": RULE_VERSION}


def extract_samples(manifest, sources, inv, root):
    selected = defaultdict(list)
    for row in manifest:
        selected[row["announcement_id"]].append(row["page"])
    pages, blocks, errors = [], [], []
    for aid, numbers in sorted(selected.items()):
        source = sources[aid]
        extractor = Extractor(source)
        started = time.monotonic()
        try:
            if inv[aid].get("error"):
                raise ValueError(inv[aid]["error"])
            with pdfplumber.open(root / source.file) as pdf:
                for number in sorted(numbers):
                    page = pdf.pages[number - 1]
                    lap = time.monotonic()
                    try:
                        extracted = extractor.page(page)
                    except Exception as error:
                        errors.append(skip(source, number, None, "P03_页解析失败", str(error)))
                        pages.append(page_record(source, number, "运行失败", time.monotonic() - lap))
                        blocks.extend(body_fragments(source, extractor.body))
                        extractor = Extractor(source)
                    else:
                        pages.append(page_record(source, number, extracted.status, time.monotonic() - lap,
                                                 table_count=len(extracted.tables), candidates=extracted.candidates,
                                                 accepted=extracted.accepted, finance_hint=extracted.finance_hint))
                        blocks.extend(extracted.blocks)
                        errors.extend(skip(source, number, *issue) for issue in extracted.skipped)
                    finally:
                        page.close()
                blocks.extend(body_fragments(source, extractor.body))
        except Exception as error:
            seen = {row["page"] for row in pages if row["announcement_id"] == aid}
            for number in sorted(set(numbers) - seen):
                errors.append(skip(source, number, None, "P04_公告读取失败", str(error)))
                pages.append(page_record(source, number, "运行失败", time.monotonic() - started))
    for block in blocks:
        block.update(parser=PARSER, rule_version=RULE_VERSION)
    return pages, blocks, errors


def page_record(source, number, status, elapsed, *, table_count=0, candidates=0, accepted=0, finance_hint=False):
    return {"announcement_id": source.id, "stock_code": source.code, "company": source.company,
            "template": ("年报" if source.period.endswith("年度") and not source.period.endswith("半年度")
                         else source.period[5:]) if source.category == "定期报告" else source.category,
            "sha256": source.sha256, "page": number, "status": status, "tables": table_count,
            "candidate_rows": candidates, "accepted_rows": accepted, "finance_hint": finance_hint,
            "fallback_attempts": 0, "seconds": round(elapsed, 4)}


def metric_rows(blocks):
    result = defaultdict(lambda: defaultdict(list))
    for block in blocks:
        if block["kind"] != "表格行" or not any(
            field["role"] == "数值" and field["unit"] in {"元", "千元", "万元", "百万元", "亿元", "人民币元"}
            for field in block["fields"]
        ):
            continue
        aid = block["announcement_id"]
        for name, pattern in METRICS.items():
            if re.fullmatch(pattern, compact(block["metric"]["text"])):
                result[aid][name].append(block["value_page"])
    return result


def verify_results(manifest, sources, inv, pages, blocks):
    expected = {(row["announcement_id"], row["page"]) for row in manifest}
    visited = {(row["announcement_id"], row["page"]) for row in pages}
    checks = {"sample_pages_accounted": visited == expected and len(pages) == len(visited),
              "fragment_citations_checked": len(blocks), "citation_errors": []}
    for index, block in enumerate(blocks, start=1):
        aid = block["announcement_id"]
        source = sources[aid]
        locations = [loc["page"] for loc in block["locations"]]
        referenced = set(locations)
        row_fields = block.get("fields", ())
        fields_ok = True
        for field in row_fields:
            referenced.update((field["header"]["page"], field["value"]["page"]))
            fields_ok = fields_ok and field["value"]["page"] == block["value_page"] and bool(field["header"]["text"])
            if field["role"] != "说明":
                referenced.add(field["unit_source"]["page"])
                fields_ok = fields_ok and bool(field["unit"] and field["unit_source"]["text"])
        if (not locations or (min(locations), max(locations)) != (block["page_start"], block["page_end"])
            or not fields_ok or block["sha256"] != source.sha256 or block["stock_code"] != source.code
            or block["source_url"] != source.url or any(
                (aid, number) not in visited or not 1 <= number <= len(inv[aid]["pages"]) for number in referenced
            )):
            checks["citation_errors"].append({"record": index, "announcement_id": aid})
    return checks


def make_report(sources, inv, pages, blocks, errors, elapsed):
    metrics = metric_rows(blocks)
    observed = defaultdict(list)
    for row in pages:
        observed[row["company"], row["template"]].append(row)
    groups = []
    for (company, template), rows in sorted(observed.items()):
        ids = {row["announcement_id"] for row in rows}
        counts = Counter(error["rule"] for error in errors if error["announcement_id"] in ids
                         and any(row["announcement_id"] == error["announcement_id"]
                                 and row["page"] == error["page"] for row in rows))
        candidates = sum(row["candidate_rows"] for row in rows)
        accepted = sum(row["accepted_rows"] for row in rows)
        groups.append({"company": company, "template": template, "sample_pages": len(rows),
                       "pages_with_tables": sum(row["tables"] > 0 for row in rows),
                       "candidate_rows": candidates, "accepted_rows": accepted,
                       "skipped_rows": candidates - accepted,
                       "suspected_financial_pages_without_candidates": sum(
                           row["finance_hint"] and not row["candidate_rows"] for row in rows),
                       "fallback_attempts": sum(row["fallback_attempts"] for row in rows),
                       "parse_seconds": round(sum(row["seconds"] for row in rows), 3),
                       "skip_reasons": counts})
    selected = {page["announcement_id"] for page in pages}
    coverage = [
        {"company": source.company, "period": source.period, "announcement_id": source.id,
         "version": source.version, "sampled": source.id in selected, "source_url": source.url,
         "metrics": {name: sorted(set(metrics.get(source.id, {}).get(name, ()))) for name in METRICS}}
        for source in sources.values() if source.category == "定期报告"
    ]
    checks = []
    for as_of, expected in (("2025-11-01", "1224774300"), ("2026-05-01", "1225273123")):
        chosen = select_version(sources.values(), "000858", "2025年三季度", "定期报告", as_of)
        aid = chosen.id if chosen else None
        found = metrics.get(aid, {})
        checks.append({"as_of": as_of, "expected_id": expected, "selected_id": aid,
                       "source_url": chosen.url if chosen else None, "passed": aid == expected,
                       "missing_metrics": [name for name in METRICS if name not in found],
                       "metric_pages": {name: sorted(set(locations)) for name, locations in found.items()}})
    totals = Counter(page["status"] for page in pages)
    records = [page for document in inv.values() for page in document["pages"]]
    return {
        "parser": PARSER, "rule_version": RULE_VERSION, "seconds": round(elapsed, 3),
        "integrity": {"announcements": len(sources), "verified_pdfs": sum(not doc.get("error") for doc in inv.values()),
                      "physical_pages": len(records), "inventory_page_errors": sum("error" in page for page in records),
                      "document_errors": {aid: doc["error"] for aid, doc in inv.items() if doc.get("error")}},
        "sample_status": totals, "unsampled_pages": len(records) - len(pages),
        "fragments": Counter(block["kind"] for block in blocks),
        "cross_page_table_rows": sum(block.get("header_page") != block.get("value_page") for block in blocks),
        "skip_reasons": Counter(error["rule"] for error in errors), "by_company_and_template": groups,
        "metric_coverage": coverage, "version_checks": checks,
        "accuracy": "未知",
        "coverage_note": "空指标列表表示当前样本没有通过规则的证据；sampled=false 表示该公告没有选取样页。",
        "fallback_note": "本轮备用解析器触发次数为零，P01 记录 OCR 候选，表格跳过记录供后续分析。",
    }


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--index", type=Path, default=INDEX)
    cli.add_argument("--output", type=Path, default=None)
    args = cli.parse_args(argv)
    root = args.index.resolve().parent
    output = args.output or root / "rag"
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    sources = read_index(args.index)
    inv = collect_inventory(sources, root)
    print(f"已核对 {len(inv)} 份索引记录，发现 {sum(len(doc['pages']) for doc in inv.values())} 页", flush=True)
    write_records(output / "inventory.jsonl", inv.values())
    path = output / "sample_manifest.jsonl"
    if path.exists():
        manifest = read_jsonl(path)
    else:
        manifest = create_manifest(sources, inv, root)
    validate_manifest(manifest, sources, inv)
    if not path.exists():
        write_records(path, manifest)
    pages, blocks, errors = extract_samples(manifest, sources, inv, root)
    for filename, records in (("sample_pages.jsonl", pages), ("fragments.jsonl", blocks), ("skips.jsonl", errors)):
        write_records(output / filename, records)
    report = make_report(sources, inv, pages, blocks, errors, time.monotonic() - started)
    checks = verify_results(manifest, sources, inv, pages, blocks)
    report["source_checks"] = checks
    write_records(output / "report.json", report, jsonl=False)
    print(f"已处理 {len(pages)} 张固定样页，生成 {len(blocks)} 个片段。报告位于 {output / 'report.json'}")
    integrity = report["integrity"]
    return int(bool(integrity["document_errors"] or integrity["inventory_page_errors"]
                    or report["sample_status"].get("运行失败")
                    or not checks["sample_pages_accounted"] or checks["citation_errors"]
                    or not all(check["passed"] for check in report["version_checks"])))


if __name__ == "__main__":
    sys.exit(main())

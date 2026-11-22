"""Parse and audit the complete Wuliangye pilot corpus."""
import argparse
import hashlib
import json
import time
from pathlib import Path

from .source import INDEX, OUTPUT, STOCK, read_index
from .validate import collect_inventory, extract_samples, make_report, verify_results, write_records

def fragment_id(block):
    anchors = [block[key] for key in ("announcement_id", "sha256", "parser", "rule_version",
                                     "kind", "locations", "text")]
    return hashlib.sha256(json.dumps(anchors, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def run(index, output):
    started = time.monotonic()
    sources = {key: source for key, source in read_index(index).items() if source.code == STOCK}
    if not sources:
        raise ValueError("索引没有五粮液公告")
    root = index.resolve().parent
    output.mkdir(parents=True, exist_ok=True)
    inv = collect_inventory(sources, root)
    manifest = [{"announcement_id": aid, "page": page["page"]}
                for aid, entry in inv.items() for page in entry["pages"]]
    print(f"已清点 {len(sources)} 份公告、{len(manifest)} 页，开始解析", flush=True)
    pages, blocks, errors = extract_samples(manifest, sources, inv, root)
    for block in blocks:
        block["id"] = fragment_id(block)
    report = make_report(sources, inv, pages, blocks, errors, time.monotonic() - started)
    report["source_checks"] = verify_results(manifest, sources, inv, pages, blocks)
    report["duplicate_fragment_ids"] = len(blocks) - len({block["id"] for block in blocks})
    checks = report["source_checks"]
    integrity = report["integrity"]
    report["passed"] = bool(checks["sample_pages_accounted"] and not checks["citation_errors"]
                            and not integrity["document_errors"] and not integrity["inventory_page_errors"]
                            and not report["sample_status"].get("运行失败")
                            and not report["duplicate_fragment_ids"] and blocks
                            and all(check["passed"] and not check["missing_metrics"]
                                    for check in report["version_checks"]))
    for name, records in (("inventory", inv.values()), ("pages", pages),
                          ("fragments", blocks), ("skips", errors)):
        write_records(output / f"{name}.jsonl", records)
    report["fragment_file_sha256"] = hashlib.sha256((output / "fragments.jsonl").read_bytes()).hexdigest()
    write_records(output / "report.json", report, jsonl=False)
    print(f"已处理 {len(pages)} 页，生成 {len(blocks)} 个片段，验收报告写入 {output / 'report.json'}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=INDEX)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    return 0 if run(args.index, args.output)["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

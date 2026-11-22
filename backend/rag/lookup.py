"""Return report figures with the source version visible on a given date."""
import argparse
import json
import re
from datetime import date
from decimal import Decimal
from pathlib import Path

from .ingest import load_audited_blocks
from .source import INDEX, OUTPUT, STOCK, read_index, select_version
from .tables import METRICS, compact

HEADERS = {
    "本期": {"本报告期", "本期", "本年"},
    "累计": {"年初至报告期末", "年初至本报告期末"},
}


def lookup(sources, blocks, period, metric, scope, as_of, *, stock=STOCK):
    source = select_version(sources, stock, period, "定期报告", as_of)
    if source is None:
        return {"status": "截止日无该报告"}
    result = {"announcement_id": source.id, "title": source.title, "disclosed_on": source.date,
              "version": source.version, "as_of": as_of, "period": period, "metric": metric,
              "scope": scope, "source_url": source.url}
    headers = HEADERS[scope]
    if scope == "本期" and period.endswith(("年度", "半年度")):
        headers = headers | {period, period[:5]}
    figures = {}
    for block in blocks:
        if (block["announcement_id"] != source.id or block["sha256"] != source.sha256
            or block["kind"] != "表格行"):
            continue
        if not re.fullmatch(METRICS[metric], compact(block["metric"]["text"])):
            continue
        if "主要会计数据" not in compact(block["section"]):
            continue
        for field in block["fields"]:
            if field["role"] == "数值" and compact(field["header"]["text"]) in headers:
                value, unit = field["value"]["text"], field["unit"]
                figures.setdefault((Decimal(compact(value).replace(",", "").replace("−", "-")), unit), []).append({
                    "value": value, "unit": unit, "header": field["header"],
                    "value_source": field["value"], "unit_source": field["unit_source"],
                    "fragment_id": block.get("id"),
                })
    result["evidence"] = [item for group in figures.values() for item in group]
    result["status"] = ("找到" if len(figures) == 1 else
                        "同一口径出现多个数值" if figures else "所选版本缺少通过校验的数字")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=INDEX)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--period", required=True)
    parser.add_argument("--as-of", required=True, type=date.fromisoformat)
    parser.add_argument("--metric", choices=METRICS, default="营业收入")
    parser.add_argument("--scope", choices=HEADERS, default="本期")
    args = parser.parse_args()
    _, _, blocks = load_audited_blocks(args.output)
    result = lookup(read_index(args.index).values(), blocks.values(), args.period, args.metric,
                    args.scope, args.as_of.isoformat())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "找到" else 1


if __name__ == "__main__":
    raise SystemExit(main())
